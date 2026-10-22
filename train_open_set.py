"""Train the fixed-taxonomy physics-conditioned Open-Set model."""

import argparse
import copy
import json
import os
import random
from typing import Dict

import numpy as np
import torch

from open_set_data import prepare_trento_open_set_dataloaders
from open_set_losses import OpenSetEnergyLoss
from open_set_metrics import compute_open_set_metrics
from open_set_model import create_open_set_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device) for name, value in batch.items()}


def forward_batch(model, batch):
    return model(batch["hsi"], batch["auxiliary"], batch["wavelengths"], batch["band_mask"])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    totals, count = {}, 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        losses = criterion(forward_batch(model, batch), batch["label"])
        losses["total"].backward()
        optimizer.step()
        size = len(batch["label"])
        count += size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    labels, closed_predictions, scores = [], [], []
    diagnostics = {name: 0.0 for name in (
        "boundary_agreement", "complementary_conflict", "harmful_conflict", "hsi_weight", "aux_weight"
    )}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = forward_batch(model, batch)
        size = len(batch["label"])
        count += size
        labels.append(batch["label"].cpu().numpy())
        closed_predictions.append(outputs["closed_set_prediction"].cpu().numpy())
        scores.append(outputs["unknown_score"].cpu().numpy())
        for name in diagnostics:
            diagnostics[name] += float(outputs[name].mean()) * size
    return (
        np.concatenate(labels), np.concatenate(closed_predictions), np.concatenate(scores),
        {name: value / count for name, value in diagnostics.items()},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Physics-conditioned multimodal Open-Set training")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--wavelengths", default=None)
    parser.add_argument("--config", default="configs/open_set_trento.json")
    parser.add_argument("--allow-linear-debug-wavelengths", action="store_true")
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--train-per-raw-class", type=int, default=125)
    parser.add_argument("--val-per-raw-class", type=int, default=50)
    parser.add_argument("--test-per-raw-class", type=int, default=500)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--output-dir", default="experiments/open_set_trento_pcme")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.5 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be between 0.5 and 1.0")
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, config, split_summary = prepare_trento_open_set_dataloaders(
        data_dir=args.data_dir,
        config_path=args.config,
        wavelength_path=args.wavelengths,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        seed=args.seed,
        augment=not args.no_augmentation,
        train_per_raw_class=args.train_per_raw_class,
        val_per_raw_class=args.val_per_raw_class,
        test_per_raw_class=args.test_per_raw_class,
        allow_linear_debug_wavelengths=args.allow_linear_debug_wavelengths,
    )
    num_known = config["num_known_classes"]
    model = create_open_set_model(
        num_known, hsi_embed_dim=args.feature_dim, feature_dim=args.feature_dim, aux_channels=1
    ).to(device)
    criterion = OpenSetEnergyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)

    history, best_state, best_val_accuracy = [], None, -1.0
    print("Open-Set spatial split:", split_summary)
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        val_labels, val_predictions, val_scores, val_diagnostics = collect_outputs(model, val_loader, device)
        val_accuracy = float(np.mean(val_labels == val_predictions))
        history.append({
            "epoch": epoch, "train": train_losses,
            "validation_known_accuracy": val_accuracy,
            "validation_unknown_score_mean": float(val_scores.mean()),
            "validation_diagnostics": val_diagnostics,
        })
        print("Epoch {:03d} | loss {:.4f} | val known {:.2f}%".format(epoch, train_losses["total"], 100 * val_accuracy))
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_labels, val_predictions, val_scores, _ = collect_outputs(model, val_loader, device)
    threshold = float(np.quantile(val_scores, args.threshold_quantile))
    test_labels, test_closed_predictions, test_scores, test_diagnostics = collect_outputs(model, test_loader, device)
    test_predictions = []
    model.eval()
    for batch in test_loader:
        batch = move_batch(batch, device)
        outputs = forward_batch(model, batch)
        test_predictions.append(model.predict_open_set(outputs, threshold).cpu().numpy())
    test_predictions = np.concatenate(test_predictions)
    test_metrics = compute_open_set_metrics(
        test_labels, test_predictions, test_scores, num_known, test_closed_predictions
    )

    class_names = config["known_class_names"] + [config["unknown_class_name"]]
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(), "threshold": threshold,
        "class_names": class_names, "open_set_config": config, "args": vars(args),
    }, checkpoint_path)
    results = {
        "class_names": class_names,
        "split_summary": split_summary,
        "threshold": threshold,
        "threshold_protocol": "known-validation quantile; no unknown test labels used",
        "best_validation_known_accuracy": best_val_accuracy,
        "test": test_metrics,
        "test_diagnostics": test_diagnostics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print("\nOpen-Set test")
    print("  known:   {:.2f}%".format(100 * test_metrics["known_accuracy"]))
    print("  unknown: {:.2f}%".format(100 * test_metrics["unknown_recall"]))
    print("  H:       {:.2f}%".format(100 * test_metrics["harmonic_mean"]))
    print("  AUROC:   {:.2f}%".format(100 * test_metrics["auroc_unknown"]))
    print("  threshold: {:.6f}".format(threshold))
    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()

