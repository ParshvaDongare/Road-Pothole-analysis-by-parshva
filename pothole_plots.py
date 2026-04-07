import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# Theme configuration — dark professional style
# ═══════════════════════════════════════════════════════════════════════

THEME = {
    "bg": "#0D1117",
    "card_bg": "#161B22",
    "text": "#E6EDF3",
    "text_muted": "#8B949E",
    "grid": "#21262D",
    "accent_orange": "#F0883E",
    "accent_green": "#3FB950",
    "accent_red": "#F85149",
    "accent_blue": "#58A6FF",
    "accent_purple": "#BC8CFF",
    "accent_yellow": "#D29922",
    "accent_cyan": "#39D2C0",
    "accent_pink": "#F778BA",
    "low": "#3FB950",
    "medium": "#D29922",
    "high": "#F85149",
    "critical": "#FF4757",
}

SEVERITY_COLORS = {
    "Low": THEME["low"],
    "Medium": THEME["medium"],
    "High": THEME["high"],
}

def apply_dark_theme():
    """Apply the dark theme to all matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["card_bg"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["text"],
        "axes.titleweight": "bold",
        "text.color": THEME["text"],
        "xtick.color": THEME["text_muted"],
        "ytick.color": THEME["text_muted"],
        "grid.color": THEME["grid"],
        "grid.alpha": 0.5,
        "legend.facecolor": THEME["card_bg"],
        "legend.edgecolor": THEME["grid"],
        "legend.labelcolor": THEME["text"],
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": THEME["bg"],
        "savefig.pad_inches": 0.3,
    })

apply_dark_theme()

# ═══════════════════════════════════════════════════════════════════════
# 1. TRAINING PERFORMANCE PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_training_loss(epochs, train_loss, val_loss, output_path="plots/01_training_loss.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, train_loss, color=THEME["accent_orange"], linewidth=2.2, label="Training loss", marker="o", markersize=4, markerfacecolor=THEME["accent_orange"])
    ax.plot(epochs, val_loss, color=THEME["accent_blue"], linewidth=2.2, label="Validation loss", marker="s", markersize=4, markerfacecolor=THEME["accent_blue"])
    ax.fill_between(epochs, train_loss, alpha=0.08, color=THEME["accent_orange"])
    ax.fill_between(epochs, val_loss, alpha=0.08, color=THEME["accent_blue"])
    best_epoch = epochs[np.argmin(val_loss)]
    best_val = min(val_loss)
    ax.axvline(x=best_epoch, color=THEME["accent_green"], linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate(f"Best: epoch {best_epoch}\\nVal loss: {best_val:.4f}", xy=(best_epoch, best_val), xytext=(best_epoch + 2, best_val + 0.01), fontsize=9, color=THEME["accent_green"], arrowprops=dict(arrowstyle="->", color=THEME["accent_green"], lw=1.2))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(framealpha=0.8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(epochs[0], epochs[-1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_map_curves(epochs, map50, map50_95, output_path="plots/02_map_curves.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, map50, color=THEME["accent_green"], linewidth=2.2, label="mAP@50", marker="o", markersize=4)
    ax.plot(epochs, map50_95, color=THEME["accent_cyan"], linewidth=2.2, label="mAP@50:95", marker="s", markersize=4)
    ax.fill_between(epochs, map50, alpha=0.08, color=THEME["accent_green"])
    ax.fill_between(epochs, map50_95, alpha=0.08, color=THEME["accent_cyan"])
    final_map50 = map50[-1]
    final_map95 = map50_95[-1]
    ax.annotate(f"Final: {final_map50:.3f}", xy=(epochs[-1], final_map50), xytext=(epochs[-1] - 5, final_map50 + 0.03), fontsize=9, color=THEME["accent_green"], arrowprops=dict(arrowstyle="->", color=THEME["accent_green"], lw=1))
    ax.annotate(f"Final: {final_map95:.3f}", xy=(epochs[-1], final_map95), xytext=(epochs[-1] - 5, final_map95 + 0.03), fontsize=9, color=THEME["accent_cyan"], arrowprops=dict(arrowstyle="->", color=THEME["accent_cyan"], lw=1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Mean Average Precision Over Training")
    ax.legend(framealpha=0.8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(epochs[0], epochs[-1])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_precision_recall(precision, recall, ap_score=None, output_path="plots/03_precision_recall.png"):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, color=THEME["accent_orange"], linewidth=2.5)
    ax.fill_between(recall, precision, alpha=0.12, color=THEME["accent_orange"])
    if ap_score is not None:
        ax.text(0.55, 0.95, f"AP = {ap_score:.3f}", transform=ax.transAxes, fontsize=16, fontweight="bold", color=THEME["accent_orange"], bbox=dict(boxstyle="round,pad=0.5", facecolor=THEME["card_bg"], edgecolor=THEME["accent_orange"], alpha=0.9))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Pothole Class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.plot([0, 1], [1, 0], linestyle="--", color=THEME["text_muted"], alpha=0.3, linewidth=1)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════
# 2. DETECTION QUALITY PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(tp, fp, fn, tn=None, output_path="plots/04_confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    if tn is not None:
        matrix = np.array([[tp, fn], [fp, tn]])
        labels = ["Pothole", "Background"]
    else:
        matrix = np.array([[tp, fn], [fp, 0]])
        labels = ["Pothole", "No Detection"]
    total = matrix.sum()
    norm_matrix = matrix / max(total, 1) * 100
    cmap_colors = [THEME["card_bg"], THEME["accent_orange"]]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom", cmap_colors, N=256)
    im = ax.imshow(norm_matrix, cmap=cmap, aspect="equal", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            pct = norm_matrix[i, j]
            color = THEME["text"] if pct > 40 else THEME["text_muted"]
            ax.text(j, i, f"{value}\\n({pct:.1f}%)", ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix")
    for spine in ax.spines.values():
        spine.set_color(THEME["grid"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percentage (%)")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_iou_distribution(iou_scores, output_path="plots/05_iou_distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, 1, 21)
    counts, _, patches = ax.hist(iou_scores, bins=bins, edgecolor=THEME["card_bg"], linewidth=0.8, alpha=0.85)
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0.5:
            patch.set_facecolor(THEME["accent_green"])
        elif left_edge >= 0.3:
            patch.set_facecolor(THEME["accent_yellow"])
        else:
            patch.set_facecolor(THEME["accent_red"])
    mean_iou = np.mean(iou_scores)
    median_iou = np.median(iou_scores)
    ax.axvline(x=mean_iou, color=THEME["accent_cyan"], linestyle="--", linewidth=1.5, label=f"Mean IoU: {mean_iou:.3f}")
    ax.axvline(x=median_iou, color=THEME["accent_purple"], linestyle=":", linewidth=1.5, label=f"Median IoU: {median_iou:.3f}")
    ax.axvline(x=0.5, color=THEME["text_muted"], linestyle="-", linewidth=1, alpha=0.5, label="IoU=0.5 threshold")
    ax.set_xlabel("IoU Score")
    ax.set_ylabel("Number of Detections")
    ax.set_title("Mask IoU Distribution (Predicted vs Ground Truth)")
    ax.legend(framealpha=0.8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    good_pct = (np.array(iou_scores) >= 0.5).sum() / len(iou_scores) * 100
    ax.text(0.97, 0.95, f"{good_pct:.0f}% above 0.5", transform=ax.transAxes, fontsize=12, fontweight="bold", color=THEME["accent_green"], ha="right", va="top", bbox=dict(boxstyle="round,pad=0.4", facecolor=THEME["card_bg"], edgecolor=THEME["accent_green"], alpha=0.9))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_confidence_distribution(confidences, fp_confidences=None, output_path="plots/06_confidence_distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, 1, 31)
    ax.hist(confidences, bins=bins, alpha=0.75, color=THEME["accent_green"], edgecolor=THEME["card_bg"], linewidth=0.5, label="True Positives")
    if fp_confidences is not None and len(fp_confidences) > 0:
        ax.hist(fp_confidences, bins=bins, alpha=0.6, color=THEME["accent_red"], edgecolor=THEME["card_bg"], linewidth=0.5, label="False Positives")
    ax.axvline(x=0.45, color=THEME["accent_yellow"], linestyle="--", linewidth=1.5, label="Confidence threshold (0.45)")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Detection Confidence Distribution")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════
# 3. SEVERITY ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_severity_distribution(severity_counts, image_names=None, output_path="plots/07_severity_distribution.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    n_images = len(severity_counts)
    if image_names is None:
        image_names = [f"Image {i+1}" for i in range(n_images)]
    x = np.arange(n_images)
    bar_width = 0.6
    low_counts = [sc.get("Low", 0) for sc in severity_counts]
    med_counts = [sc.get("Medium", 0) for sc in severity_counts]
    high_counts = [sc.get("High", 0) for sc in severity_counts]
    bars_low = ax.bar(x, low_counts, bar_width, label="Low", color=THEME["low"], alpha=0.85, edgecolor=THEME["card_bg"])
    bars_med = ax.bar(x, med_counts, bar_width, bottom=low_counts, label="Medium", color=THEME["medium"], alpha=0.85, edgecolor=THEME["card_bg"])
    bottoms_high = [l + m for l, m in zip(low_counts, med_counts)]
    bars_high = ax.bar(x, high_counts, bar_width, bottom=bottoms_high, label="High", color=THEME["high"], alpha=0.85, edgecolor=THEME["card_bg"])
    for i in range(n_images):
        total = low_counts[i] + med_counts[i] + high_counts[i]
        ax.text(i, total + 0.15, str(total), ha="center", va="bottom", fontsize=11, fontweight="bold", color=THEME["text"])
    ax.set_xlabel("Test Images")
    ax.set_ylabel("Number of Potholes")
    ax.set_title("Severity Distribution Across Test Images")
    ax.set_xticks(x)
    ax.set_xticklabels(image_names, rotation=30, ha="right", fontsize=9)
    ax.legend(framealpha=0.8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_area_vs_depth_scatter(pothole_data_list, output_path="plots/08_area_vs_depth.png"):
    fig, ax = plt.subplots(figsize=(10, 7))
    for severity, color in SEVERITY_COLORS.items():
        subset = [p for p in pothole_data_list if p.get("severity") == severity]
        if not subset:
            continue
        areas = [p["area_ratio"] * 100 for p in subset]
        depths = [p.get("normalized_depth", 0) for p in subset]
        scores = [p.get("severity_score", 0.5) for p in subset]
        sizes = [max(30, s * 200) for s in scores]
        ax.scatter(areas, depths, c=color, s=sizes, alpha=0.7, edgecolors="white", linewidth=0.5, label=f"{severity}", zorder=3)
    ax.set_xlabel("Area (% of image)")
    ax.set_ylabel("Normalized Depth")
    ax.set_title("Pothole Area vs Depth — Size = Severity Score")
    ax.axhline(y=0.5, color=THEME["text_muted"], linestyle=":", alpha=0.4)
    ax.axvline(x=2.0, color=THEME["text_muted"], linestyle=":", alpha=0.4)
    ax.text(0.5, 0.12, "Small & Shallow\\n(Low risk)", transform=ax.transAxes, fontsize=9, color=THEME["text_muted"], ha="center", alpha=0.6)
    ax.text(0.85, 0.85, "Large & Deep\\n(High risk)", transform=ax.transAxes, fontsize=9, color=THEME["accent_red"], ha="center", alpha=0.7)
    ax.legend(framealpha=0.8, loc="upper left", title="Severity")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_rdi_radar(rdi_components, image_name="Test Image", output_path="plots/09_rdi_radar.png"):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["card_bg"])
    categories = ["Count\\nFactor", "Area\\nFactor", "Severity\\nFactor", "Spatial\\nFactor"]
    values = [
        rdi_components.get("count_factor", 0),
        rdi_components.get("area_factor", 0),
        rdi_components.get("severity_factor", 0),
        rdi_components.get("spatial_factor", 0),
    ]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax.plot(angles, values, "o-", linewidth=2.5, color=THEME["accent_orange"], markersize=8, markerfacecolor=THEME["accent_orange"])
    ax.fill(angles, values, alpha=0.15, color=THEME["accent_orange"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color=THEME["text"])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=THEME["text_muted"])
    ax.grid(True, alpha=0.3, color=THEME["grid"])
    rdi = rdi_components.get("rdi", 0)
    rdi_color = (THEME["accent_green"] if rdi < 0.3 else THEME["accent_yellow"] if rdi < 0.5 else THEME["accent_red"])
    ax.set_title(f"Road Damage Index — {image_name}\\nRDI = {rdi:.3f}", fontsize=14, fontweight="bold", color=THEME["text"], pad=24)
    circle = plt.Circle((0.5, 0.02), 0.06, transform=ax.transAxes, color=rdi_color, alpha=0.2, zorder=0)
    fig.gca().add_patch(circle)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════
# 4. SEGMENTATION QUALITY PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_shape_metrics_distribution(shape_data, output_path="plots/10_shape_metrics.png"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics = [
        ("circularity", "Circularity", THEME["accent_blue"]),
        ("solidity", "Solidity", THEME["accent_green"]),
        ("roughness", "Roughness", THEME["accent_orange"]),
        ("convexity_deficit", "Convexity Deficit", THEME["accent_purple"]),
    ]
    for ax, (key, label, color) in zip(axes, metrics):
        values = [s.get(key, 0) for s in shape_data if s.get(key) is not None]
        if not values:
            values = [0]
        bp = ax.boxplot(values, vert=True, patch_artist=True, widths=0.5, boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color), whiskerprops=dict(color=color, linewidth=1.2), capprops=dict(color=color, linewidth=1.2), medianprops=dict(color=THEME["text"], linewidth=2), flierprops=dict(markeredgecolor=color, markersize=4))
        jittered_x = np.random.normal(1, 0.04, len(values))
        ax.scatter(jittered_x, values, c=color, alpha=0.5, s=20, zorder=3, edgecolors="none")
        ax.set_title(label, fontsize=11, color=color)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis="y")
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color=color, linestyle=":", alpha=0.5, linewidth=1)
        ax.text(1.35, mean_val, f"μ={mean_val:.2f}", fontsize=8, color=color, va="center")
    fig.suptitle("Shape Metrics Distribution Across All Detected Potholes", fontsize=14, fontweight="bold", color=THEME["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_segmentation_comparison(v1_ious, v3_ious, output_path="plots/11_segmentation_comparison.png"):
    fig, ax = plt.subplots(figsize=(9, 6))
    data = [v1_ious, v3_ious]
    positions = [1, 2]
    colors = [THEME["accent_red"], THEME["accent_green"]]
    vp = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.6)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.3)
        body.set_edgecolor(colors[i])
    vp["cmeans"].set_color(THEME["text"])
    vp["cmedians"].set_color(THEME["accent_cyan"])
    vp["cmins"].set_color(THEME["text_muted"])
    vp["cmaxes"].set_color(THEME["text_muted"])
    vp["cbars"].set_color(THEME["text_muted"])
    for i, (d, pos, col) in enumerate(zip(data, positions, colors)):
        jx = np.random.normal(pos, 0.05, len(d))
        ax.scatter(jx, d, c=col, alpha=0.4, s=15, zorder=3, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(["v1 — Bounding Box\\nSegmentation", "v3 — Shape\\nSegmentation"], fontsize=11)
    ax.set_ylabel("Mask IoU Score")
    ax.set_title("Segmentation Quality: v1 (Rectangle) vs v3 (Contour-Aware)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color=THEME["text_muted"], linestyle=":", alpha=0.4, linewidth=1)
    improvement = np.mean(v3_ious) - np.mean(v1_ious)
    sign = "+" if improvement > 0 else ""
    ax.text(0.5, 0.95, f"Mean IoU improvement: {sign}{improvement:.3f}", transform=ax.transAxes, fontsize=12, fontweight="bold", color=THEME["accent_green"] if improvement > 0 else THEME["accent_red"], ha="center", va="top", bbox=dict(boxstyle="round,pad=0.4", facecolor=THEME["card_bg"], edgecolor=THEME["accent_green"] if improvement > 0 else THEME["accent_red"], alpha=0.9))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════
# 5. ROAD CONDITION SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def plot_multi_image_summary(image_summaries, output_path="plots/12_road_summary.png"):
    n = len(image_summaries)
    fig = plt.figure(figsize=(16, 5 + n * 0.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    names = [s["name"] for s in image_summaries]
    counts = [s["n_potholes"] for s in image_summaries]
    colors_count = [THEME["accent_green"] if c <= 3 else THEME["accent_yellow"] if c <= 6 else THEME["accent_red"] for c in counts]
    bars = ax1.barh(names, counts, color=colors_count, alpha=0.8, edgecolor=THEME["card_bg"])
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=10, color=THEME["text"])
    ax1.set_xlabel("Count")
    ax1.set_title("Potholes per image")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.invert_yaxis()
    ax2 = fig.add_subplot(gs[0, 1])
    rdis = [s["rdi"] for s in image_summaries]
    colors_rdi = [THEME["accent_green"] if r < 0.3 else THEME["accent_yellow"] if r < 0.5 else THEME["accent_red"] for r in rdis]
    bars2 = ax2.barh(names, rdis, color=colors_rdi, alpha=0.8, edgecolor=THEME["card_bg"])
    for bar, rdi in zip(bars2, rdis):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{rdi:.2f}", va="center", fontsize=10, color=THEME["text"])
    ax2.set_xlabel("RDI Score")
    ax2.set_title("Road Damage Index")
    ax2.set_xlim(0, 1.0)
    ax2.axvline(x=0.3, color=THEME["accent_yellow"], linestyle=":", alpha=0.5)
    ax2.axvline(x=0.5, color=THEME["accent_red"], linestyle=":", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()
    ax3 = fig.add_subplot(gs[0, 2])
    conditions = [s["road_condition"] for s in image_summaries]
    cond_colors = {"Good": THEME["accent_green"], "Fair": THEME["accent_cyan"], "Moderate": THEME["accent_yellow"], "Poor": THEME["accent_orange"], "Critical": THEME["accent_red"]}
    for i, (name, cond) in enumerate(zip(names, conditions)):
        color = cond_colors.get(cond, THEME["text_muted"])
        ax3.text(0.5, i, cond.upper(), ha="center", va="center", fontsize=14, fontweight="bold", color=color, bbox=dict(boxstyle="round,pad=0.4", facecolor=THEME["card_bg"], edgecolor=color, alpha=0.9))
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(names)
    ax3.set_xlim(0, 1)
    ax3.set_title("Road condition grade")
    ax3.set_xticks([])
    ax3.invert_yaxis()
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax4 = fig.add_subplot(gs[1, 0:2])
    area_pcts = [s["total_area_pct"] for s in image_summaries]
    x_pos = np.arange(n)
    ax4.bar(x_pos, area_pcts, color=THEME["accent_orange"], alpha=0.8, edgecolor=THEME["card_bg"], width=0.5)
    for i, pct in enumerate(area_pcts):
        ax4.text(i, pct + 0.1, f"{pct:.1f}%", ha="center", fontsize=9, color=THEME["accent_orange"])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax4.set_ylabel("Total damaged area (%)")
    ax4.set_title("Total road surface affected")
    ax4.grid(True, alpha=0.3, axis="y")
    ax5 = fig.add_subplot(gs[1, 2])
    total_low = sum(s.get("low_count", 0) for s in image_summaries)
    total_med = sum(s.get("medium_count", 0) for s in image_summaries)
    total_high = sum(s.get("high_count", 0) for s in image_summaries)
    sizes = [total_low, total_med, total_high]
    labels_pie = [f"Low ({total_low})", f"Medium ({total_med})", f"High ({total_high})"]
    colors_pie = [THEME["low"], THEME["medium"], THEME["high"]]
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax5.pie(sizes, labels=labels_pie, colors=colors_pie, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9, "color": THEME["text"]}, wedgeprops={"edgecolor": THEME["card_bg"], "linewidth": 1.5})
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
            at.set_color(THEME["text"])
    ax5.set_title("Overall severity split")
    fig.suptitle("Road Condition Analysis — Multi-Image Summary", fontsize=16, fontweight="bold", color=THEME["text"], y=1.01)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_fp_filter_comparison(before_counts, after_counts, image_names=None, output_path="plots/13_fp_filter_effect.png"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = len(before_counts)
    if image_names is None:
        image_names = [f"Image {i+1}" for i in range(n)]
    x = np.arange(n)
    width = 0.35
    bars_before = ax.bar(x - width/2, before_counts, width, label="Before FP filter", color=THEME["accent_red"], alpha=0.7, edgecolor=THEME["card_bg"])
    bars_after = ax.bar(x + width/2, after_counts, width, label="After FP filter", color=THEME["accent_green"], alpha=0.7, edgecolor=THEME["card_bg"])
    for i in range(n):
        removed = before_counts[i] - after_counts[i]
        if removed > 0:
            ax.annotate(f"-{removed} FP", xy=(x[i], max(before_counts[i], after_counts[i]) + 0.3), fontsize=9, color=THEME["accent_yellow"], ha="center", fontweight="bold")
    ax.set_xlabel("Test Images")
    ax.set_ylabel("Detection Count")
    ax.set_title("False Positive Filtering Effect (v3)")
    ax.set_xticks(x)
    ax.set_xticklabels(image_names, rotation=30, ha="right", fontsize=9)
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    total_before = sum(before_counts)
    total_after = sum(after_counts)
    reduction = (1 - total_after / max(total_before, 1)) * 100
    ax.text(0.97, 0.95, f"FP reduction: {reduction:.0f}%", transform=ax.transAxes, fontsize=12, fontweight="bold", color=THEME["accent_green"], ha="right", va="top", bbox=dict(boxstyle="round,pad=0.4", facecolor=THEME["card_bg"], edgecolor=THEME["accent_green"], alpha=0.9))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════
# DEMO: Generate all plots with realistic sample data
# ═══════════════════════════════════════════════════════════════════════

def generate_sample_data():
    np.random.seed(42)
    data = {}
    epochs = list(range(1, 31))
    base_train_loss = 0.12 * np.exp(-0.08 * np.array(epochs)) + 0.025
    data["epochs"] = epochs
    data["train_loss"] = (base_train_loss + np.random.normal(0, 0.003, 30)).tolist()
    data["val_loss"] = (base_train_loss * 1.15 + np.random.normal(0, 0.005, 30)).tolist()
    map50_base = 1 - 0.65 * np.exp(-0.12 * np.array(epochs))
    data["map50"] = np.clip(map50_base + np.random.normal(0, 0.015, 30), 0, 1).tolist()
    data["map50_95"] = np.clip(map50_base * 0.72 + np.random.normal(0, 0.012, 30), 0, 1).tolist()
    recall = np.linspace(0, 1, 100)
    precision = np.clip(1 - 0.2 * recall - 0.5 * recall**3 + np.random.normal(0, 0.01, 100), 0, 1)
    precision = np.sort(precision)[::-1]
    data["recall"] = recall.tolist()
    data["precision"] = precision.tolist()
    data["ap_score"] = float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2))
    data["tp"] = 47
    data["fp"] = 8
    data["fn"] = 5
    data["tn"] = 140
    data["iou_scores"] = np.clip(np.concatenate([np.random.beta(5, 2, 40), np.random.beta(2, 3, 10)]), 0, 1).tolist()
    data["tp_confidences"] = np.clip(np.random.beta(8, 2, 47), 0.3, 1).tolist()
    data["fp_confidences"] = np.clip(np.random.beta(2, 4, 8), 0.1, 0.7).tolist()
    data["image_names"] = ["potholes042", "potholes065", "potholes108", "potholes134", "potholes201", "potholes287"]
    data["severity_counts"] = [{"Low": 2, "Medium": 1, "High": 2}, {"Low": 1, "Medium": 0, "High": 0}, {"Low": 2, "Medium": 1, "High": 1}, {"Low": 3, "Medium": 2, "High": 1}, {"Low": 0, "Medium": 1, "High": 2}, {"Low": 4, "Medium": 1, "High": 0}]
    pothole_data = []
    for sc in data["severity_counts"]:
        for sev, count in sc.items():
            for _ in range(count):
                if sev == "Low":
                    ar = np.random.uniform(0.002, 0.015)
                    nd = np.random.uniform(0.0, 0.4)
                elif sev == "Medium":
                    ar = np.random.uniform(0.01, 0.05)
                    nd = np.random.uniform(0.3, 0.7)
                else:
                    ar = np.random.uniform(0.03, 0.12)
                    nd = np.random.uniform(0.5, 1.0)
                ss = 0.5 * (ar / 0.10) + 0.3 * nd + 0.2 * np.random.uniform(0, 0.3)
                pothole_data.append({"area_ratio": ar, "normalized_depth": nd, "severity": sev, "severity_score": ss})
    data["pothole_scatter"] = pothole_data
    data["rdi_components"] = {"count_factor": 0.4, "area_factor": 0.55, "severity_factor": 0.72, "spatial_factor": 0.6, "rdi": 0.558}
    data["shape_metrics"] = [{"circularity": np.random.uniform(0.2, 0.8), "solidity": np.random.uniform(0.6, 0.95), "roughness": np.random.uniform(1.0, 1.6), "convexity_deficit": np.random.uniform(0.05, 0.35)} for _ in range(24)]
    data["v1_ious"] = np.clip(np.random.beta(3, 4, 50), 0, 1).tolist()
    data["v3_ious"] = np.clip(np.random.beta(6, 2.5, 50), 0, 1).tolist()
    data["image_summaries"] = [{"name": "potholes042", "n_potholes": 5, "rdi": 0.62, "road_condition": "Poor", "total_area_pct": 8.2, "max_severity": "High", "high_count": 2, "medium_count": 1, "low_count": 2}, {"name": "potholes065", "n_potholes": 1, "rdi": 0.08, "road_condition": "Good", "total_area_pct": 0.6, "max_severity": "Low", "high_count": 0, "medium_count": 0, "low_count": 1}, {"name": "potholes108", "n_potholes": 4, "rdi": 0.55, "road_condition": "Poor", "total_area_pct": 6.3, "max_severity": "High", "high_count": 1, "medium_count": 1, "low_count": 2}, {"name": "potholes134", "n_potholes": 6, "rdi": 0.71, "road_condition": "Critical", "total_area_pct": 11.5, "max_severity": "High", "high_count": 1, "medium_count": 2, "low_count": 3}, {"name": "potholes201", "n_potholes": 3, "rdi": 0.58, "road_condition": "Poor", "total_area_pct": 5.4, "max_severity": "High", "high_count": 2, "medium_count": 1, "low_count": 0}, {"name": "potholes287", "n_potholes": 5, "rdi": 0.32, "road_condition": "Moderate", "total_area_pct": 3.1, "max_severity": "Medium", "high_count": 0, "medium_count": 1, "low_count": 4}]
    data["before_fp"] = [12, 3, 14, 9, 7, 8]
    data["after_fp"] = [5, 1, 4, 6, 3, 5]
    return data

def generate_all_plots(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    data = generate_sample_data()
    print("\\n" + "=" * 60)
    print("GENERATING SHOWCASE PLOTS")
    print("=" * 60)
    print("\\n── Training Performance ──")
    plot_training_loss(data["epochs"], data["train_loss"], data["val_loss"], f"{output_dir}/01_training_loss.png")
    plot_map_curves(data["epochs"], data["map50"], data["map50_95"], f"{output_dir}/02_map_curves.png")
    plot_precision_recall(data["precision"], data["recall"], data["ap_score"], f"{output_dir}/03_precision_recall.png")
    print("\\n── Detection Quality ──")
    plot_confusion_matrix(data["tp"], data["fp"], data["fn"], data["tn"], f"{output_dir}/04_confusion_matrix.png")
    plot_iou_distribution(data["iou_scores"], f"{output_dir}/05_iou_distribution.png")
    plot_confidence_distribution(data["tp_confidences"], data["fp_confidences"], f"{output_dir}/06_confidence_distribution.png")
    print("\\n── Severity Analysis ──")
    plot_severity_distribution(data["severity_counts"], data["image_names"], f"{output_dir}/07_severity_distribution.png")
    plot_area_vs_depth_scatter(data["pothole_scatter"], f"{output_dir}/08_area_vs_depth.png")
    plot_rdi_radar(data["rdi_components"], "potholes108", f"{output_dir}/09_rdi_radar.png")
    print("\\n── Segmentation Quality ──")
    plot_shape_metrics_distribution(data["shape_metrics"], f"{output_dir}/10_shape_metrics.png")
    plot_segmentation_comparison(data["v1_ious"], data["v3_ious"], f"{output_dir}/11_segmentation_comparison.png")
    print("\\n── Road Condition Summary ──")
    plot_multi_image_summary(data["image_summaries"], f"{output_dir}/12_road_summary.png")
    plot_fp_filter_comparison(data["before_fp"], data["after_fp"], data["image_names"], f"{output_dir}/13_fp_filter_effect.png")
    print(f"\\n{'='*60}")
    print(f"All 13 plots saved to: {output_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    generate_all_plots()
