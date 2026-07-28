"""Evaluate a released HAANet checkpoint on a reproducible HAA10K split.

The released ``val.py`` behavior is preserved where it affects the metrics:
an 80/20 entity split, five random left/right view draws per validation entity,
and accuracy computed over individual draws. This entry point additionally
fixes all random seeds, reports entity-level accuracy after prediction
averaging, and saves inspectable JSON/CSV results.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import model3_enhanced
from dataset import HairFaceDataset
from model3_enhanced import HairFaceAestheticModel


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PAPER_METRICS = {
    "lcc": 0.781,
    "srcc": 0.742,
    "mse": 0.824,
    "accuracy": 0.682,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_DIR / "dataset" / "images",
    )
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
        "--model-weights",
        type=Path,
        default=PROJECT_DIR / "model" / "best_aesthetic_model3.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "evaluation_results",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-multiplier", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit validation entities after splitting (for pipeline smoke tests).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def score_level(value: float) -> int:
    if 0 <= value < 2:
        return 0
    if 2 <= value < 4:
        return 1
    if 4 <= value < 6:
        return 2
    if 6 <= value < 8:
        return 3
    if 8 <= value <= 10:
        return 4
    return -1


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_split(
    data_root: Path,
    labels_path: Path,
    test_size: float,
    seed: int,
) -> list[dict]:
    with labels_path.open("r", encoding="utf-8") as stream:
        labels = json.load(stream)

    data_info = []
    for sample_id, values in labels.items():
        data_info.append(
            {
                "sample_id": sample_id,
                "images": [
                    str(data_root / f"{sample_id}_{view}.png")
                    for view in range(12)
                ],
                "score": float(values["score"]),
            }
        )

    _, validation_info = train_test_split(
        data_info,
        test_size=test_size,
        random_state=seed,
    )
    return validation_info


def validate_paths(data_info: list[dict]) -> None:
    missing = [
        path
        for info in data_info
        for path in info["images"]
        if not Path(path).is_file()
    ]
    if missing:
        preview = missing[:10]
        raise FileNotFoundError(
            f"{len(missing)} validation images are missing. First paths: {preview}"
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model3_enhanced.device = device
    validation_info = load_split(
        args.data_root,
        args.labels,
        args.test_size,
        args.seed,
    )
    if args.max_samples is not None:
        validation_info = validation_info[: args.max_samples]
    validate_paths(validation_info)

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
    dataset = HairFaceDataset(
        data_info=validation_info,
        transform=transform,
        augment_multiplier=args.augment_multiplier,
        random_seed=args.seed,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )

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

    predictions: dict[str, list[float]] = defaultdict(list)
    truths: dict[str, float] = {}
    pass_correct = 0
    pass_total = 0
    started = time.time()

    with torch.inference_mode():
        for batch_index, (views, scores, _) in enumerate(
            tqdm(loader, desc="HAANet evaluation")
        ):
            start = batch_index * args.batch_size
            stop = start + len(scores)
            entity_indices = dataset.data_indices[start:stop]
            sample_ids = [
                dataset.data_info[index]["sample_id"] for index in entity_indices
            ]

            views = views.to(device, non_blocking=True)
            with contextlib.redirect_stdout(io.StringIO()):
                predicted_scores, _ = model(views, views[:, 0])
            predicted_values = predicted_scores.detach().cpu().numpy()
            true_values = scores.numpy()

            for sample_id, prediction, truth in zip(
                sample_ids,
                predicted_values,
                true_values,
            ):
                prediction_value = float(prediction)
                truth_value = float(truth)
                predictions[sample_id].append(prediction_value)
                truths[sample_id] = truth_value
                pass_correct += score_level(prediction_value) == score_level(
                    truth_value
                )
                pass_total += 1

    rows = []
    for sample_id in sorted(predictions, key=int):
        prediction = float(np.mean(predictions[sample_id]))
        truth = truths[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "true_score": truth,
                "predicted_score": prediction,
                "absolute_error": abs(prediction - truth),
                "true_level": score_level(truth),
                "predicted_level": score_level(prediction),
                "num_draws": len(predictions[sample_id]),
            }
        )

    true_scores = np.asarray([row["true_score"] for row in rows])
    predicted_scores = np.asarray([row["predicted_score"] for row in rows])
    lcc = float(pearsonr(true_scores, predicted_scores).statistic)
    srcc = float(spearmanr(true_scores, predicted_scores).statistic)
    errors = predicted_scores - true_scores
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    released_accuracy = pass_correct / pass_total
    entity_accuracy = float(
        np.mean(
            [
                row["true_level"] == row["predicted_level"]
                for row in rows
            ]
        )
    )

    metrics = {
        "protocol": {
            "split": f"{1 - args.test_size:.0%}/{args.test_size:.0%}",
            "split_seed": args.seed,
            "validation_entities": len(rows),
            "view_draws_per_entity": args.augment_multiplier,
            "inference_passes": pass_total,
            "batch_size": args.batch_size,
            "note": (
                "Single released checkpoint; not the paper's five-fold average."
            ),
        },
        "metrics": {
            "lcc": lcc,
            "srcc": srcc,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "accuracy_released_code": released_accuracy,
            "accuracy_entity_averaged": entity_accuracy,
        },
        "paper_table_1": PAPER_METRICS,
        "delta_vs_paper": {
            "lcc": lcc - PAPER_METRICS["lcc"],
            "srcc": srcc - PAPER_METRICS["srcc"],
            "mse": mse - PAPER_METRICS["mse"],
            "accuracy_released_code": (
                released_accuracy - PAPER_METRICS["accuracy"]
            ),
            "accuracy_entity_averaged": (
                entity_accuracy - PAPER_METRICS["accuracy"]
            ),
        },
        "runtime_seconds": time.time() - started,
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / (
        f"metrics_seed{args.seed}_n{len(rows)}_a{args.augment_multiplier}.json"
    )
    rows_path = args.output_dir / (
        f"predictions_seed{args.seed}_n{len(rows)}_a{args.augment_multiplier}.csv"
    )
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics_path={metrics_path}")
    print(f"predictions_path={rows_path}")


if __name__ == "__main__":
    main()
