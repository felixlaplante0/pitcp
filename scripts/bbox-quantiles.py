import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import OxfordIIITPet

plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

data = torch.load("../data/test_coords.pt")
all_gt = data["gt"]
all_pred = data["pred"]

dataset = OxfordIIITPet(root="./data", split="test", download=True)
all_labels = np.array([target for _, target in dataset])
class_names = dataset.classes


def iou(bboxes1, bboxes2):
    x1 = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    y1 = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    x2 = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    y2 = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1 + area2 - intersection
    return intersection / union


scores = 1 - iou(all_gt, all_pred).numpy()

plt.figure(figsize=(10, 5))
cmap = plt.cm.tab20.colors + plt.cm.tab20b.colors

all_q_values = []

for class_id in range(len(class_names)):
    mask = all_labels == class_id
    class_ious = scores[mask]

    if len(class_ious) == 0:
        continue

    sorted_class_ious = np.sort(class_ious)
    quantiles = np.linspace(0, 1, len(sorted_class_ious))

    idx_90 = int(0.9 * (len(sorted_class_ious) - 1))
    all_q_values.append(sorted_class_ious[idx_90])

    plt.plot(
        quantiles,
        sorted_class_ious,
        label=class_names[class_id],
        color=cmap[class_id],
        alpha=0.6,
    )

p_target = 0.9
val_min = min(all_q_values)
val_max = max(all_q_values)
segment_len = val_max - val_min

plt.axvline(x=p_target, color="black", linestyle="dashed", alpha=0.3)
plt.vlines(x=p_target, ymin=val_min, ymax=val_max, color="#ff0000", linewidth=2)
cap_width = 0.015
plt.hlines(
    y=[val_min, val_max],
    xmin=p_target - cap_width,
    xmax=p_target + cap_width,
    color="#ff0000",
    linewidth=2,
)
plt.text(
    p_target + 0.02,
    (val_min + val_max) / 2,
    f"{segment_len:.3f}",
    color="#ff0000",
    fontweight="bold",
    ha="left",
    va="center",
    fontsize=16,
)

plt.xlabel(r"Quantile ($1 - \alpha$)")
plt.ylabel("Nonconformity score (1 - IoU)")
plt.title("Conditional nonconformity score quantiles by pet breed")

plt.legend(loc="upper left", ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig("../figures/bbox-quantiles.pdf")
plt.show()
