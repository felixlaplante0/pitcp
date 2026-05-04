import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms import v2
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(device)

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(320, max_size=321, interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop(320),
    ]
)


def run_and_save(split, filename):
    ds = OxfordIIITPet(
        root="./data", split=split, target_types="segmentation", download=True
    )
    all_gt = []
    all_pred = []

    for i in tqdm(range(len(ds)), desc=split):
        img_pil, mask_pil = ds[i]
        img_t, mask_t = transform(img_pil, mask_pil)

        nz = torch.nonzero(mask_t[0] * 255 == 1)
        if nz.any():
            gt = torch.tensor(
                [nz[:, 1].min(), nz[:, 0].min(), nz[:, 1].max(), nz[:, 0].max()]
            )
        else:
            gt = torch.zeros(4)

        with torch.no_grad():
            out = model([img_t.to(device)])[0]

        idx = torch.argmax(out["scores"])
        pred = out["boxes"][idx].cpu()

        all_gt.append(gt)
        all_pred.append(pred)

    torch.save({"gt": torch.stack(all_gt), "pred": torch.stack(all_pred)}, filename)


run_and_save("trainval", "trainval_coords.pt")
run_and_save("test", "test_coords.pt")
