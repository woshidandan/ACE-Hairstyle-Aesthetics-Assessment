"""Validate imports, checkpoints, GPU execution, and HAANet tensor shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model3_enhanced import HairFaceAestheticModel


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hair-weights",
        type=Path,
        default=PROJECT_DIR / "model" / "hair.pt",
    )
    parser.add_argument(
        "--face-weights",
        type=Path,
        default=PROJECT_DIR / "model" / "face.pt",
    )
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=PROJECT_DIR / "model" / "best_aesthetic_model3.pth",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-full-checkpoint", action="store_true")
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="Also run one backward/optimizer step with the released losses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = HairFaceAestheticModel(
        str(args.hair_weights),
        str(args.face_weights),
        hair_dim=256,
        face_dim=256,
        hidden_dim=512,
    ).to(device)
    if not args.skip_full_checkpoint:
        model.load_state_dict(
            torch.load(
                args.model_weights,
                map_location=device,
                weights_only=True,
            )
        )
    views = torch.randn(1, 4, 3, 224, 224, device=device)
    if args.train_step:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        target_score = torch.tensor([5.0], device=device)
        target_angles = torch.tensor([[0, 6, 3, 9]], device=device)
        score, angle_logits = model(views, views[:, 0])
        regression_loss = torch.nn.functional.l1_loss(score, target_score)
        angle_loss = torch.nn.functional.cross_entropy(
            angle_logits.reshape(-1, 12),
            target_angles.reshape(-1),
        )
        rule_reg_loss = (
            torch.norm(model.rule_layer.rule_matrix_1, p=1)
            + torch.norm(model.rule_layer.rule_matrix_2, p=1)
            + torch.norm(model.rule_layer.rule_matrix_3, p=1)
        ) * 0.1
        total_loss = regression_loss + 0.5 * angle_loss + rule_reg_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        print(f"train_step_loss={float(total_loss.detach()):.6f}")
    else:
        model.eval()
        with torch.inference_mode():
            score, angle_logits = model(views, views[:, 0])

    assert score.shape == (1,), score.shape
    assert angle_logits.shape == (1, 4, 12), angle_logits.shape
    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"capability={torch.cuda.get_device_capability(device)}")
    print(f"score_shape={tuple(score.shape)}")
    print(f"angle_logits_shape={tuple(angle_logits.shape)}")
    if args.train_step:
        print("HAANet backward/optimizer step passed.")
    print("HAANet smoke test passed.")


if __name__ == "__main__":
    main()
