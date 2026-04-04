"""
Complete pothole detection pipeline in a single Python file.

This script covers:
1. Pascal VOC XML to YOLO segmentation TXT conversion using SAM when available
2. Train/validation/test split
3. YOLOv8 segmentation training with pretrained weights
4. Testing and inference
5. Pothole counting and mask-area calculation
6. MiDaS depth estimation
7. Severity classification and road-condition grading
8. Final console + visual output

Dataset expected structure in the current working directory:
    images/
    annotations/

Requirements:
    pip install ultralytics opencv-python torch torchvision matplotlib pyyaml
    pip install segment-anything

Place a SAM checkpoint such as `sam_vit_b_01ec64.pth` in the project root,
or update `SAM_CHECKPOINT` below.

Example:
    python pothole_detection_pipeline.py
"""

import os
import random
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_CACHE_DIR)
os.environ["TORCH_HOME"] = str(LOCAL_CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(LOCAL_CACHE_DIR / "matplotlib")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("Starting pothole pipeline imports...", flush=True)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import YOLO

print("Imports loaded.", flush=True)


warnings.filterwarnings("ignore")


BASE_DIR = PROJECT_ROOT
IMAGES_DIR = BASE_DIR / "images"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

WORKSPACE_DIR = BASE_DIR / "pothole_yolo_workspace"
YOLO_LABELS_DIR = WORKSPACE_DIR / "labels_yolo_seg"
SPLIT_DIR = WORKSPACE_DIR / "dataset"
TRAIN_IMAGES_DIR = SPLIT_DIR / "images" / "train"
VAL_IMAGES_DIR = SPLIT_DIR / "images" / "val"
TEST_IMAGES_DIR = SPLIT_DIR / "images" / "test"
TRAIN_LABELS_DIR = SPLIT_DIR / "labels" / "train"
VAL_LABELS_DIR = SPLIT_DIR / "labels" / "val"
TEST_LABELS_DIR = SPLIT_DIR / "labels" / "test"

YAML_PATH = WORKSPACE_DIR / "data.yaml"
RUNS_DIR = WORKSPACE_DIR / "runs"
RESULTS_DIR = WORKSPACE_DIR / "results"

CLASS_NAMES = ["pothole"]
CLASS_TO_ID = {"pothole": 0}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEFAULT_CONFIDENCE_THRESHOLD = 0.50
MASK_IOU_DUPLICATE_THRESHOLD = 0.50
MAX_POLYGON_POINTS = 48
CONVERSION_PROGRESS_INTERVAL = 25

SAM_CHECKPOINT = PROJECT_ROOT / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"


def load_sam_predictor(preferred_device=None):
    """
    Load the Segment Anything predictor once.

    If the dependency or checkpoint is missing, return None and let the
    converter fall back to rectangle polygons.
    """
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print(
            "WARNING: segment-anything is not installed.\n"
            "Run: pip install segment-anything\n"
            "Falling back to rectangle polygons for all annotations."
        )
        return None

    if not SAM_CHECKPOINT.exists():
        print(
            f"WARNING: SAM checkpoint not found at {SAM_CHECKPOINT}\n"
            "Falling back to rectangle polygons for all annotations."
        )
        return None

    if preferred_device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = preferred_device
    print(f"Loading SAM ({SAM_MODEL_TYPE}) on {device}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM loaded successfully.")
    return predictor


def get_sam_device(predictor):
    """Return the current device used by a SAM predictor."""
    if predictor is None:
        return "cpu"
    return next(predictor.model.parameters()).device.type


def is_cuda_oom(error):
    """Check whether an exception was caused by CUDA running out of memory."""
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or (
        "cuda" in message and "out of memory" in message
    )


def generate_mask_with_sam(predictor, xmin, ymin, xmax, ymax):
    """
    Use SAM to generate a mask for one pothole bounding box.

    Returns a full-image binary mask with values 0/1 or None when prediction
    fails or the mask is too small to be useful.
    """
    if predictor is None:
        return None

    box_np = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    try:
        masks, scores, _ = predictor.predict(
            box=box_np,
            multimask_output=True,
        )
    except Exception as exc:
        if is_cuda_oom(exc):
            raise
        print(f"  SAM prediction failed for box ({xmin}, {ymin}, {xmax}, {ymax}): {exc}")
        return None

    best_index = int(np.argmax(scores))
    mask = masks[best_index].astype(np.uint8)

    box_area = max(1.0, (xmax - xmin) * (ymax - ymin))
    if float(mask.sum()) < 0.01 * box_area:
        return None

    return mask


def fallback_sam_to_cpu(current_predictor, reason_text):
    """Reload SAM on CPU after a CUDA OOM so label generation can continue."""
    if current_predictor is None or get_sam_device(current_predictor) != "cuda":
        return current_predictor

    print(
        f"  WARNING: SAM ran out of GPU memory while {reason_text}. "
        "Reloading SAM on CPU and continuing label conversion.",
        flush=True,
    )
    del current_predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return load_sam_predictor(preferred_device="cpu")


def clamp(value, min_value, max_value):
    """Keep a value inside a valid range."""
    return max(min_value, min(value, max_value))


def approximate_contour(contour, max_points=MAX_POLYGON_POINTS):
    """Simplify a contour so the YOLO polygon stays compact."""
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(1.0, 0.01 * perimeter)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

    while len(simplified) > max_points and epsilon < perimeter:
        epsilon *= 1.35
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

    return simplified


def mask_to_yolo_polygon(mask, image_width, image_height):
    """
    Convert a binary mask into a YOLO segmentation polygon.

    Returns a flat list [x1, y1, x2, y2, ...] with normalized coordinates.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    simplified = approximate_contour(contour)
    if len(simplified) < 3:
        return None

    points = []
    for point in simplified[:, 0, :]:
        points.append(clamp(float(point[0]) / image_width, 0.0, 1.0))
        points.append(clamp(float(point[1]) / image_height, 0.0, 1.0))

    return points if len(points) >= 6 else None


def bbox_to_rectangle_polygon(width, height, xmin, ymin, xmax, ymax):
    """Fallback polygon when SAM is unavailable or fails."""
    points = [
        (xmin / width, ymin / height),
        (xmax / width, ymin / height),
        (xmax / width, ymax / height),
        (xmin / width, ymax / height),
    ]
    return [clamp(value, 0.0, 1.0) for point in points for value in point]


def create_folders():
    """Create all directories required for the pipeline."""
    folders = [
        WORKSPACE_DIR,
        YOLO_LABELS_DIR,
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TEST_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR,
        TEST_LABELS_DIR,
        RESULTS_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def clear_split_directories():
    """Clean previous split files so the script can be rerun safely."""
    folders = [
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TEST_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR,
        TEST_LABELS_DIR,
    ]
    for folder in folders:
        if folder.exists():
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()


def validate_dataset_paths():
    """Check that dataset folders exist and contain files."""
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"Annotations folder not found: {ANNOTATIONS_DIR}")
    if not list(IMAGES_DIR.glob("*")):
        raise FileNotFoundError(f"No image files found in: {IMAGES_DIR}")
    if not list(ANNOTATIONS_DIR.glob("*.xml")):
        raise FileNotFoundError(f"No XML annotation files found in: {ANNOTATIONS_DIR}")


def get_image_path_for_xml(xml_file: Path):
    """Find the matching image file for a given XML file."""
    for extension in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = IMAGES_DIR / f"{xml_file.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def copy_pair_to_split(image_path, label_path, image_dest, label_dest):
    """Copy an image/label pair into the selected split folder."""
    shutil.copy2(image_path, image_dest / image_path.name)
    shutil.copy2(label_path, label_dest / label_path.name)


def convert_xml_to_yolo(sam_predictor):
    """
    Convert Pascal VOC XML boxes into YOLO segmentation TXT labels.

    The converter first tries SAM to get a pothole-shaped polygon and falls
    back to a rectangle polygon if SAM is unavailable or fails.
    """
    print("\nConverting Pascal VOC XML to YOLO segmentation labels (SAM-powered)...")
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    converted_pairs = []
    sam_success = 0
    rectangle_fallback = 0
    total_files = len(xml_files)

    for file_index, xml_file in enumerate(xml_files, start=1):
        if (
            file_index == 1
            or file_index % CONVERSION_PROGRESS_INTERVAL == 0
            or file_index == total_files
        ):
            print(
                f"  [{file_index}/{total_files}] SAM masks: {sam_success} | "
                f"rectangle fallbacks: {rectangle_fallback}",
                flush=True,
            )

        image_path = get_image_path_for_xml(xml_file)
        if image_path is None:
            print(f"  Skipping {xml_file.name}: no matching image found.")
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"  Skipping {xml_file.name}: image could not be read.")
            continue

        image_height, image_width = image_bgr.shape[:2]

        image_rgb = None
        if sam_predictor is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            try:
                sam_predictor.set_image(image_rgb)
            except Exception as exc:
                if is_cuda_oom(exc):
                    sam_predictor = fallback_sam_to_cpu(
                        sam_predictor,
                        f"preparing {xml_file.name}",
                    )
                    if sam_predictor is not None:
                        sam_predictor.set_image(image_rgb)
                else:
                    raise

        tree = ET.parse(xml_file)
        root = tree.getroot()

        size_node = root.find("size")
        if size_node is not None:
            xml_width = int(size_node.find("width").text)
            xml_height = int(size_node.find("height").text)
            if xml_width != image_width or xml_height != image_height:
                print(
                    f"  NOTE {xml_file.name}: XML size {xml_width}x{xml_height} "
                    f"!= image size {image_width}x{image_height}. Using image size."
                )

        label_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip().lower()
            if class_name not in CLASS_TO_ID:
                continue

            bbox_node = obj.find("bndbox")
            xmin = float(bbox_node.find("xmin").text)
            ymin = float(bbox_node.find("ymin").text)
            xmax = float(bbox_node.find("xmax").text)
            ymax = float(bbox_node.find("ymax").text)

            xmin = clamp(xmin, 0, image_width - 1)
            ymin = clamp(ymin, 0, image_height - 1)
            xmax = clamp(xmax, xmin + 1, image_width)
            ymax = clamp(ymax, ymin + 1, image_height)

            polygon_points = None
            if sam_predictor is not None:
                try:
                    sam_mask = generate_mask_with_sam(
                        sam_predictor,
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    )
                except Exception as exc:
                    if is_cuda_oom(exc):
                        sam_predictor = fallback_sam_to_cpu(
                            sam_predictor,
                            f"predicting a mask in {xml_file.name}",
                        )
                        if sam_predictor is not None:
                            if image_rgb is None:
                                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                            sam_predictor.set_image(image_rgb)
                            sam_mask = generate_mask_with_sam(
                                sam_predictor,
                                xmin,
                                ymin,
                                xmax,
                                ymax,
                            )
                        else:
                            sam_mask = None
                    else:
                        raise
                if sam_mask is not None:
                    polygon_points = mask_to_yolo_polygon(
                        sam_mask,
                        image_width,
                        image_height,
                    )

            if polygon_points is None:
                polygon_points = bbox_to_rectangle_polygon(
                    image_width,
                    image_height,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                )
                rectangle_fallback += 1
            else:
                sam_success += 1

            class_id = CLASS_TO_ID[class_name]
            coords_str = " ".join(f"{value:.6f}" for value in polygon_points)
            label_lines.append(f"{class_id} {coords_str}")

        label_path = YOLO_LABELS_DIR / f"{xml_file.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as file:
            file.write("\n".join(label_lines))

        if label_lines:
            converted_pairs.append((image_path, label_path))

    print("\nConversion complete.")
    print(f"  Total label files written : {len(converted_pairs)}")
    print(f"  SAM-generated masks       : {sam_success}")
    print(f"  Rectangle fallbacks       : {rectangle_fallback}")
    return converted_pairs, sam_predictor


def split_dataset(image_label_pairs):
    """Split the dataset into train/val/test folders."""
    print("\nSplitting dataset into train/val/test...")
    clear_split_directories()
    random.shuffle(image_label_pairs)
    total_samples = len(image_label_pairs)

    train_count = int(total_samples * TRAIN_RATIO)
    val_count = int(total_samples * VAL_RATIO)

    train_pairs = image_label_pairs[:train_count]
    val_pairs = image_label_pairs[train_count : train_count + val_count]
    test_pairs = image_label_pairs[train_count + val_count :]

    for image_path, label_path in train_pairs:
        copy_pair_to_split(image_path, label_path, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    for image_path, label_path in val_pairs:
        copy_pair_to_split(image_path, label_path, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    for image_path, label_path in test_pairs:
        copy_pair_to_split(image_path, label_path, TEST_IMAGES_DIR, TEST_LABELS_DIR)

    print(f"  Train : {len(train_pairs)}")
    print(f"  Val   : {len(val_pairs)}")
    print(f"  Test  : {len(test_pairs)}")

    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
        "counts": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
            "total": total_samples,
        },
    }


def create_data_yaml():
    """Create the YOLO data.yaml file."""
    print("\nCreating data.yaml...")
    data = {
        "path": str(SPLIT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
        "nc": len(CLASS_NAMES),
    }
    with open(YAML_PATH, "w", encoding="utf-8") as file:
        yaml.dump(data, file, sort_keys=False)
    print(f"  Saved: {YAML_PATH}")
    return YAML_PATH


def train_yolov8_model(data_yaml_path):
    """Train the YOLOv8 segmentation model."""
    print("\nTraining YOLOv8 segmentation model...")
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(data_yaml_path),
        epochs=20,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR),
        name="pothole_segmenter",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    best_model_path = RUNS_DIR / "pothole_segmenter" / "weights" / "best.pt"
    print(f"  Best model: {best_model_path}")
    return best_model_path


def load_trained_model(model_path):
    """Load a trained YOLO model from disk."""
    print("\nLoading trained YOLOv8 model...")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    return YOLO(str(model_path))


def run_detection_on_test_images(model):
    """Run inference on the test split and return YOLO results."""
    print("\nRunning detection on test images...")
    test_images = sorted(TEST_IMAGES_DIR.glob("*"))
    if not test_images:
        raise FileNotFoundError("No test images found.")

    all_results = []
    for image_path in test_images:
        results = model.predict(
            source=str(image_path),
            conf=DEFAULT_CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False,
        )
        all_results.append((image_path, results[0]))

    print(f"  Processed {len(all_results)} test images.")
    return all_results


def keep_largest_component(binary_mask):
    """Keep only the largest connected component in a binary mask."""
    if binary_mask is None or np.count_nonzero(binary_mask) == 0:
        return binary_mask

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return binary_mask.astype(np.uint8)

    largest_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    output_mask[labels == largest_index] = 1
    return output_mask


def compute_mask_iou(mask_a, mask_b):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(intersection / union) if union > 0 else 0.0


def remove_duplicate_detections(pothole_data):
    """Remove heavily overlapping duplicate detections."""
    if not pothole_data:
        return pothole_data

    sorted_data = sorted(
        pothole_data,
        key=lambda item: item.get("confidence", 0.0),
        reverse=True,
    )
    filtered = []

    for candidate in sorted_data:
        duplicate_found = any(
            compute_mask_iou(candidate["mask"], kept["mask"]) >= MASK_IOU_DUPLICATE_THRESHOLD
            for kept in filtered
            if candidate.get("mask") is not None and kept.get("mask") is not None
        )
        if not duplicate_found:
            filtered.append(candidate)

    for index, pothole in enumerate(filtered, start=1):
        pothole["id"] = index

    return filtered


def extract_pothole_features(detection_result, image_shape):
    """Extract bbox, mask, area, and confidence for each pothole."""
    boxes = detection_result.boxes
    masks = detection_result.masks
    image_height, image_width = image_shape[:2]
    image_area = float(image_width * image_height)
    pothole_data = []

    if boxes is None or len(boxes) == 0:
        return pothole_data

    for idx, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = xyxy

        if masks is not None and masks.data is not None and idx < len(masks.data):
            binary_mask = masks.data[idx].cpu().numpy()
            binary_mask = (binary_mask > 0.5).astype(np.uint8)
            if binary_mask.shape != (image_height, image_width):
                binary_mask = cv2.resize(
                    binary_mask,
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            binary_mask = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8),
                iterations=1,
            )
            binary_mask = keep_largest_component(binary_mask)
            
            if np.count_nonzero(binary_mask) > 0:
                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh
                    
            area_pixels = float(np.count_nonzero(binary_mask))
        else:
            binary_mask = None
            area_pixels = float((x2 - x1) * (y2 - y1))
        pothole_data.append(
            {
                "id": idx + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "mask": binary_mask,
                "confidence": confidence,
                "area_pixels": area_pixels,
                "area_ratio": area_pixels / image_area if image_area > 0 else 0.0,
            }
        )

    return remove_duplicate_detections(pothole_data)


def load_midas_model(device):
    """Load the MiDaS depth-estimation model."""
    print("\nLoading MiDaS depth model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return midas, transforms.small_transform


def estimate_depth_map(image_bgr, midas_model, transform, device):
    """Compute a full-image relative depth map with MiDaS."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(image_rgb).to(device)

    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


def estimate_depth_for_box(depth_map, bbox, mask=None):
    """Estimate mean relative depth inside a bbox or segmentation mask."""
    image_height, image_width = depth_map.shape[:2]
    x1 = clamp(int(bbox[0]), 0, image_width - 1)
    y1 = clamp(int(bbox[1]), 0, image_height - 1)
    x2 = clamp(int(bbox[2]), x1 + 1, image_width)
    y2 = clamp(int(bbox[3]), y1 + 1, image_height)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    if mask is not None:
        mask_region = mask[y1:y2, x1:x2]
        if mask_region.size == 0 or np.count_nonzero(mask_region) == 0:
            return 0.0
        return float(np.mean(region[mask_region > 0]))

    return float(np.mean(region))


def add_depth_information(image_bgr, pothole_data, midas_model, transform, device):
    """Estimate depth for each pothole and attach normalized depth values."""
    depth_map = estimate_depth_map(image_bgr, midas_model, transform, device)
    depth_values = []

    for pothole in pothole_data:
        depth_value = estimate_depth_for_box(depth_map, pothole["bbox"], pothole.get("mask"))
        pothole["raw_depth"] = depth_value
        depth_values.append(depth_value)

    if depth_values:
        min_depth = min(depth_values)
        max_depth = max(depth_values)
        depth_span = max_depth - min_depth if max_depth - min_depth > 1e-6 else 1.0
        for pothole in pothole_data:
            pothole["normalized_depth"] = float((pothole["raw_depth"] - min_depth) / depth_span)

    return pothole_data, depth_map


def assign_severity_labels(pothole_data, image_width=None, image_height=None):
    """
    Assign severity using absolute image-area ratios.

    image_width and image_height are optional for compatibility with the GUI
    and dashboard. When omitted, the function tries to infer the image size
    from the first available mask.
    """
    if not pothole_data:
        return pothole_data

    if image_width is None or image_height is None:
        first_mask = next((item.get("mask") for item in pothole_data if item.get("mask") is not None), None)
        if first_mask is not None:
            image_height, image_width = first_mask.shape[:2]
        else:
            max_x = max(item["bbox"][2] for item in pothole_data)
            max_y = max(item["bbox"][3] for item in pothole_data)
            image_width = max(image_width or 1, int(max_x))
            image_height = max(image_height or 1, int(max_y))

    image_area = float(image_width * image_height)

    for pothole in pothole_data:
        area_ratio = pothole["area_pixels"] / image_area if image_area > 0 else 0.0
        
        # Smooth normalized area mapping where Large easily pushes the score up
        if area_ratio < 0.02:
            size_label = "Small"
            normalized_area = 0.2 * (area_ratio / 0.02)
        elif area_ratio < 0.08:
            size_label = "Medium"
            normalized_area = 0.2 + 0.6 * ((area_ratio - 0.02) / 0.06)
        else:
            size_label = "Large"
            normalized_area = 0.8 + 0.2 * min(1.0, (area_ratio - 0.08) / 0.08)

        normalized_depth = pothole.get("normalized_depth", 0.0)
        
        # 60% Area / 40% Depth weighted score
        severity_score = (0.6 * normalized_area) + (0.4 * normalized_depth)

        pothole["area_ratio"] = area_ratio
        pothole["normalized_area"] = normalized_area
        pothole["size_label"] = size_label
        pothole["severity_score"] = severity_score
        pothole["severity"] = (
            "Low"
            if severity_score < 0.35
            else "Medium"
            if severity_score < 0.60
            else "High"
        )

    return pothole_data


def summarize_road_condition(pothole_data):
    """Summarize the overall road condition from pothole severities."""
    if not pothole_data:
        return "Good"

    severities = [item.get("severity", "Low") for item in pothole_data]
    if "High" in severities:
        return "Poor"
    if severities.count("Medium") > len(severities) / 2:
        return "Moderate"
    return "Good"


def get_severity_color(severity):
    """Return a BGR color for each severity label."""
    return {
        "Low": (0, 255, 0),
        "Medium": (0, 255, 255),
        "High": (0, 0, 255),
    }.get(severity, (255, 255, 255))


def render_annotated_image(image_bgr, pothole_data):
    """Draw segmentation overlays, contours, and labels on the output image."""
    output_image = image_bgr.copy()

    for pothole in pothole_data:
        x1, y1 = pothole["bbox"][0], pothole["bbox"][1]
        color = get_severity_color(pothole.get("severity", "Low"))
        label = (
            f"#{pothole['id']} {pothole.get('severity', '?')} | "
            f"area:{pothole['area_ratio'] * 100:.1f}% | "
            f"d:{pothole.get('raw_depth', 0.0):.2f}"
        )

        mask = pothole.get("mask")
        if mask is not None and np.count_nonzero(mask) > 0:
            overlay = output_image.copy()
            overlay[mask > 0] = color
            output_image = cv2.addWeighted(overlay, 0.30, output_image, 0.70, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(output_image, contours, -1, color, 2)
        else:
            cv2.rectangle(
                output_image,
                (pothole["bbox"][0], pothole["bbox"][1]),
                (pothole["bbox"][2], pothole["bbox"][3]),
                color,
                2,
            )

        # Draw a black border for text readability
        cv2.putText(
            output_image,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output_image,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return output_image


def display_and_save_results(image_path, image_bgr, pothole_data):
    """Print results and display/save the final annotated image."""
    road_condition = summarize_road_condition(pothole_data)
    annotated_image = render_annotated_image(image_bgr, pothole_data)

    print("\n" + "=" * 60)
    print(f"Image             : {image_path.name}")
    print(f"Potholes detected : {len(pothole_data)}")
    print(f"Road condition    : {road_condition}")

    if not pothole_data:
        print("No road damage detected.")

    for pothole in pothole_data:
        print(f"\n  Pothole {pothole['id']}:")
        print(
            f"    Area (px / % image) : {pothole['area_pixels']:.0f} / "
            f"{pothole['area_ratio'] * 100:.2f}%"
        )
        print(f"    Size category       : {pothole.get('size_label', '?')}")
        print(
            f"    Depth (relative)    : {pothole.get('raw_depth', 0.0):.4f} "
            f"(norm: {pothole.get('normalized_depth', 0.0):.4f})"
        )
        print(f"    Severity score      : {pothole.get('severity_score', 0.0):.4f}")
        print(f"    Severity            : {pothole.get('severity', '?')}")
        print(f"    Confidence          : {pothole.get('confidence', 0.0):.2f}")

    output_path = RESULTS_DIR / f"result_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"\n  Saved: {output_path}")

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Pothole Detection - {image_path.name} | Road: {road_condition}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_pipeline():
    """Run the complete end-to-end pothole segmentation pipeline."""
    print("=" * 70)
    print("POTHOLE DETECTION PIPELINE (SAM-powered segmentation)")
    print("=" * 70)

    validate_dataset_paths()
    create_folders()

    sam_predictor = load_sam_predictor()

    image_label_pairs, sam_predictor = convert_xml_to_yolo(sam_predictor)
    if not image_label_pairs:
        raise RuntimeError("No valid image/XML pairs found. Pipeline cannot continue.")

    if sam_predictor is not None:
        del sam_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    split_info = split_dataset(image_label_pairs)
    if split_info["counts"]["test"] == 0:
        raise RuntimeError("Test split is empty. Use a larger dataset.")

    data_yaml_path = create_data_yaml()
    best_model_path = train_yolov8_model(data_yaml_path)
    yolo_model = load_trained_model(best_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model, midas_transform = load_midas_model(device)

    detection_results = run_detection_on_test_images(yolo_model)
    for image_path, detection_result in detection_results:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image_height, image_width = image_bgr.shape[:2]
        pothole_data = extract_pothole_features(detection_result, image_bgr.shape)
        pothole_data, _ = add_depth_information(
            image_bgr,
            pothole_data,
            midas_model,
            midas_transform,
            device,
        )
        pothole_data = assign_severity_labels(pothole_data, image_width, image_height)
        display_and_save_results(image_path, image_bgr, pothole_data)

    print("\nPipeline completed successfully.")
    print(f"  Trained model : {best_model_path}")
    print(f"  Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    run_pipeline()
