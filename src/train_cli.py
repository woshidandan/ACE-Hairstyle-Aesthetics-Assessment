"""Parameterized training entry point for the released HAANet implementation.

This wrapper preserves the released model, losses, optimizer, and default Rule
Layer update schedule. It replaces hard-coded paths and hyperparameters with
command-line arguments and adds bounded diagnostics that write to explicit
checkpoint locations.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import model3_enhanced
import training
from dataset import HairFaceDataset, load_data
from model3_enhanced import HairFaceAestheticModel
from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_DIR / "HAA10K" / "score.json",
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
        "--save-path",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--last-epoch-path",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "last_epoch.pth",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_DIR / "runs",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--augment-multiplier", type=int, default=5)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training entities after splitting (for safe smoke tests).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Limit validation entities after splitting (for safe smoke tests).",
    )
    parser.add_argument(
        "--skip-centroid-update",
        action="store_true",
        help="Skip the epoch-0/10/... K-Means update (diagnostics only).",
    )
    parser.add_argument(
        "--skip-last-epoch-save",
        action="store_true",
        help="Do not write the additional last-epoch checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.augment_multiplier < 1:
        raise ValueError("--augment-multiplier must be at least 1.")
    if (
        not args.skip_centroid_update
        and args.max_train_samples is not None
        and args.max_train_samples < 12
    ):
        raise ValueError(
            "--max-train-samples must be at least 12 when K-Means centroid "
            "updates are enabled."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model3_enhanced.device = device
    training.device = device

    missing = [
        str(path)
        for path in (args.data_root, args.labels, args.hair_weights, args.face_weights)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Required paths not found: {missing}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    train_info, val_info = load_data(
        str(args.data_root),
        str(args.labels),
        test_size=args.test_size,
        random_state=args.seed,
    )
    if args.max_train_samples is not None:
        train_info = train_info[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_info = val_info[: args.max_val_samples]

    print(
        "Training configuration: "
        f"train_entities={len(train_info)}, "
        f"val_entities={len(val_info)}, "
        f"augment_multiplier={args.augment_multiplier}, "
        f"best_output={args.save_path}, "
        f"last_output="
        f"{None if args.skip_last_epoch_save else args.last_epoch_path}"
    )
    train_dataset = HairFaceDataset(
        train_info,
        transform=transform,
        augment_multiplier=args.augment_multiplier,
        random_seed=args.seed,
    )
    val_dataset = HairFaceDataset(
        val_info,
        transform=transform,
        augment_multiplier=args.augment_multiplier,
        random_seed=args.seed,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    model = HairFaceAestheticModel(
        str(args.hair_weights),
        str(args.face_weights),
        hair_dim=256,
        face_dim=256,
        hidden_dim=512,
    ).to(device)
    regression_criterion = nn.L1Loss()
    angle_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_last_epoch_save:
        args.last_epoch_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    try:
        train(
            model,
            train_loader,
            val_loader,
            regression_criterion,
            angle_criterion,
            optimizer,
            scheduler,
            writer,
            args.epochs,
            str(args.save_path),
            (
                None
                if args.skip_last_epoch_save
                else str(args.last_epoch_path)
            ),
            not args.skip_centroid_update,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
