import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.ops import box_iou
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes

data = torch.load("../data/test_coords.pt")
all_gt = data["gt"]
all_pred = data["pred"]

dataset = OxfordIIITPet(
    root="./data", split="test", target_types="segmentation", download=True
)

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(320, max_size=321, interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop(320),
    ]
)

num_samples = 5
np.random.seed(50)
idxs = np.random.choice(np.arange(len(dataset)), num_samples)

fig, axs = plt.subplots(1, num_samples, figsize=(20, 5))

for i, idx in enumerate(idxs):
    img_pil, _ = dataset[idx]
    img_tensor = transform(img_pil)

    gt_box = all_gt[idx]
    pred_box = all_pred[idx]

    iou_matrix = box_iou(gt_box.unsqueeze(0), pred_box.unsqueeze(0))
    current_iou = iou_matrix.item()

    boxes = [gt_box, pred_box]
    labels = ["GT", "Pred"]
    colors = ["#00ff00", "#ff0000"]

    res = draw_bounding_boxes(
        img_tensor,
        torch.stack(boxes),
        labels=labels,
        colors=colors,
        width=5,
        font="LiberationSans-Regular",
        font_size=25,
    )

    axs[i].imshow(res.permute(1, 2, 0))
    axs[i].set_title(f"IoU: {current_iou:.4f}", fontsize=20)
    axs[i].axis("off")

plt.tight_layout()
plt.savefig("../figures/bbox-visu.pdf")
plt.show()
