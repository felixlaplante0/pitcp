"""Regenerates the confidence-independent playground model artifact."""

from pathlib import Path

import numpy as np
import torch
import zuko

from pitcp import CONTRA, HPD, PITCP
from playground.utils import SEED, _data

ROOT = Path(__file__).resolve().parent


def main():  # noqa: D103
    data = _data()
    torch.manual_seed(SEED)
    score_density = zuko.flows.SOSPF(features=1, context=1, hidden_features=(32, 32))
    pitcp = PITCP(
        score_density,
        torch.optim.Adam(score_density.parameters(), lr=1e-3),
        n_epochs=200,
        batch_size=512,
        verbose=True,
    )
    pitcp.fit(data["x_train"], np.abs(data["y_train"])).conformalize(
        data["x_cal"], np.abs(data["y_cal"])
    )
    torch.save(
        {
            "state_dict": pitcp.state_dict(),
            "scores": torch.as_tensor(pitcp.scores_),
        },
        ROOT / "models" / "pitcp_example.pt",
    )

    torch.manual_seed(SEED)
    density = zuko.flows.MAF(features=1, context=1, hidden_features=(32, 32))
    optimizer = torch.optim.Adam(density.parameters(), lr=1e-3)
    hpd = HPD(
        density,
        optimizer,
        n_epochs=500,
        n_samples=1000,
        batch_size=512,
        verbose=True,
        random_state=SEED,
    )
    hpd.fit(data["x_train"], data["y_train"]).conformalize(data["x_cal"], data["y_cal"])
    contra = CONTRA(density, optimizer, batch_size=512, verbose=False)
    contra.conformalize(data["x_cal"], data["y_cal"])
    torch.save(
        {
            "state_dict": density.state_dict(),
            "hpd_scores": torch.as_tensor(hpd.scores_),
            "contra_scores": torch.as_tensor(contra.scores_),
        },
        ROOT / "models" / "density_example.pt",
    )


if __name__ == "__main__":
    main()
