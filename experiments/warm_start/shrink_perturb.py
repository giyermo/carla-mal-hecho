"""
Shrink & Perturb (Ash & Adams, "On Warm-Starting Neural Network Training",
NeurIPS 2020, section 4).

Before training a new round, every learnable weight of last round's model is
moved toward zero and given a little random noise:

    theta_new = lam * theta_old + noise_scale * theta_random

theta_random comes from a freshly initialized copy of the same network, as in
the paper. Shrinking keeps what the model learned but makes it less confident,
so old and new data both produce useful gradients again. The noise helps the
model train faster afterwards.

Usage:
    python shrink_perturb.py runs/.../weights/last.pt sp.pt --lam 0.6 --noise-scale 0.01
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def shrink_perturb(
    weights: str | Path,
    out_path: str | Path,
    lam: float = 0.6,
    noise_scale: float = 0.01,
    model_cfg: str = "yolo26n.yaml",
    seed: int = 0,
) -> Path:
    """
    Apply Shrink & Perturb to a trained YOLO checkpoint and save the result.

    weights:     last round's checkpoint (e.g. weights/last.pt)
    out_path:    where to save the new starting checkpoint
    lam:         shrink factor, 0 < lam < 1 (paper's main setting: 0.6)
    noise_scale: size of the noise (paper: 0.01)
    model_cfg:   architecture, used to build the randomly initialized network
    seed:        makes the noise reproducible
    """
    if not 0 < lam < 1:
        raise ValueError(f"lam must be between 0 and 1, got {lam}")

    model = YOLO(str(weights))
    trained = model.model  # the PyTorch network inside the YOLO wrapper

    # A fresh, randomly initialized network with the same architecture and
    # number of classes. This is the "perturb" part.
    torch.manual_seed(seed)
    random_net = DetectionModel(model_cfg, nc=trained.nc, verbose=False)
    random_params = dict(random_net.named_parameters())

    with torch.no_grad():
        # named_parameters() contains only the learnable weights. BatchNorm's
        # running mean/variance are buffers (statistics of the data, not learned
        # by gradient descent), so they are left unchanged.
        for name, param in trained.named_parameters():
            noise = random_params[name]
            if noise.shape != param.shape:
                raise ValueError(f"{name}: shape {tuple(param.shape)} vs {tuple(noise.shape)}")
            param.mul_(lam).add_(noise.to(param.dtype), alpha=noise_scale)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Shrink & Perturb to a YOLO checkpoint.")
    parser.add_argument("weights", type=Path, help="last round's checkpoint")
    parser.add_argument("out", type=Path, help="output checkpoint")
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--model-cfg", default="yolo26n.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = shrink_perturb(args.weights, args.out, args.lam, args.noise_scale, args.model_cfg, args.seed)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
