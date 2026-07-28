"""Run HAANet inference on one four-view portrait sample.

The view order follows the released training code:
front (angle 0), back (angle 6), one left view, and one right view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model3_enhanced import HairFaceAestheticModel


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--sample-dir",
        type=Path,
        help="Directory containing <sample_id>_0.png through <sample_id>_11.png.",
    )
    source.add_argument("--front", type=Path, help="Explicit frontal-view image.")
    parser.add_argument("--sample-id", help="Sample prefix used with --sample-dir.")
    parser.add_argument("--back", type=Path, help="Explicit back-view image.")
    parser.add_argument("--left", type=Path, help="Explicit left-view image.")
    parser.add_argument("--right", type=Path, help="Explicit right-view image.")
    parser.add_argument(
        "--left-index",
        type=int,
        default=3,
        choices=range(1, 6),
        help="Left-view index for dataset-layout input (default: 3).",
    )
    parser.add_argument(
        "--right-index",
        type=int,
        default=9,
        choices=range(7, 12),
        help="Right-view index for dataset-layout input (default: 9).",
    )
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
    return parser.parse_args()


def resolve_views(args: argparse.Namespace) -> list[Path]:
    if args.sample_dir is not None:
        if not args.sample_id:
            raise SystemExit("--sample-id is required with --sample-dir.")
        indices = (0, 6, args.left_index, args.right_index)
        paths = [args.sample_dir / f"{args.sample_id}_{index}.png" for index in indices]
    else:
        if not all((args.back, args.left, args.right)):
            raise SystemExit("--back, --left and --right are required with --front.")
        paths = [args.front, args.back, args.left, args.right]

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input images not found: {missing}")
    return paths


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    view_paths = resolve_views(args)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    views = torch.stack(
        [transform(Image.open(path).convert("RGB")) for path in view_paths]
    ).unsqueeze(0).to(device)

    model = HairFaceAestheticModel(
        str(args.hair_weights),
        str(args.face_weights),
        hair_dim=256,
        face_dim=256,
        hidden_dim=512,
    ).to(device)
    state_dict = torch.load(
        args.model_weights,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.inference_mode():
        score, angle_logits = model(views, views[:, 0])

    result = {
        "score": float(score.item()),
        "view_paths": [str(path.resolve()) for path in view_paths],
        "predicted_angle_classes": angle_logits.argmax(dim=-1)[0].tolist(),
        "device": str(device),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
