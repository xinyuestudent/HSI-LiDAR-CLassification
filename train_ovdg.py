"""Single-domain OVDG training entry point used before multi-source training."""

import argparse
import copy
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch

from clip_prompt_bank import load_prompt_bank
from ovdg_data import prepare_trento_ovdg_dataloaders
from ovdg_losses import OVDGLoss
from ovdg_metrics import compute_ovdg_metrics
from ovdg_model import create_ovdg_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def forward_batch(model, batch):
    return model(
        hsi=batch["hsi"],
        auxiliary=batch["auxiliary"],
        wavelengths=batch["wavelengths"],
        band_mask=batch["band_mask"],
    )


def train_epoch(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    totals = {}
    num_samples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        outputs = forward_batch(model, batch)
        losses = criterion(outputs, batch["label"])
        losses["total"].backward()
        optimizer.step()

        batch_size = batch["label"].shape[0]
        num_samples += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    return {name: value / num_samples for name, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, seen_mask) -> Tuple[Dict, Dict]:
    model.eval()
    labels, predictions = [], []
    loss_total = 0.0
    num_samples = 0
    diagnostic_names = [
        "boundary_agreement", "complementary_conflict", "harmful_conflict",
        "hsi_weight", "aux_weight",
    ]
    diagnostic_totals = {name: 0.0 for name in diagnostic_names}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = forward_batch(model, batch)
        losses = criterion(outputs, batch["label"])
        prediction = outputs["logits"].argmax(dim=-1)
        batch_size = batch["label"].shape[0]
        loss_total += float(losses["total"]) * batch_size
        num_samples += batch_size
        labels.append(batch["label"].cpu().numpy())
        predictions.append(prediction.cpu().numpy())
        for name in diagnostic_names:
            diagnostic_totals[name] += float(outputs[name].mean()) * batch_size

    metrics = compute_ovdg_metrics(
        np.concatenate(labels), np.concatenate(predictions), seen_mask
    )
    metrics["loss"] = loss_total / num_samples
    diagnostics = {
        name: value / num_samples for name, value in diagnostic_totals.items()
    }
    return metrics, diagnostics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the first seen-to-open-vocabulary OVDG-Mamba stage"
    )
    parser.add_argument("--dataset", default="trento", choices=["trento"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prompt-bank", required=True)
    parser.add_argument("--wavelengths", default=None)
    parser.add_argument(
        "--mapping", default="configs/ovdg_trento.json",
        help="Original-label to canonical Prompt Bank mapping",
    )
    parser.add_argument("--allow-linear-debug-wavelengths", action="store_true")
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--output-dir", default="experiments/ovdg_trento_stage1")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    bank = load_prompt_bank(args.prompt_bank)
    train_loader, val_loader, test_loader = prepare_trento_ovdg_dataloaders(
        data_dir=args.data_dir,
        prompt_bank=bank,
        mapping_path=args.mapping,
        wavelength_path=args.wavelengths,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        seed=args.seed,
        augment=not args.no_augmentation,
        allow_linear_debug_wavelengths=args.allow_linear_debug_wavelengths,
    )

    model = create_ovdg_model(
        text_embeddings=bank["embeddings"],
        num_seen_classes=bank["num_seen_classes"],
        hsi_embed_dim=args.feature_dim,
        feature_dim=args.feature_dim,
        aux_channels=1,
    ).to(device)
    criterion = OVDGLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    seen_mask = bank["seen_mask"].cpu().numpy().astype(bool)
    os.makedirs(args.output_dir, exist_ok=True)

    best_state = None
    best_seen_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_diagnostics = evaluate(
            model, val_loader, criterion, device, seen_mask
        )
        record = {
            "epoch": epoch,
            "train": train_losses,
            "validation": val_metrics,
            "validation_diagnostics": val_diagnostics,
        }
        history.append(record)
        print(
            "Epoch {:03d} | loss {:.4f} | val seen {:.2f}%".format(
                epoch,
                train_losses["total"],
                100.0 * val_metrics["seen_accuracy"],
            )
        )
        if val_metrics["seen_accuracy"] > best_seen_accuracy:
            best_seen_accuracy = val_metrics["seen_accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics, test_diagnostics = evaluate(
        model, test_loader, criterion, device, seen_mask
    )
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": bank["class_names"],
            "seen_mask": bank["seen_mask"],
            "prompt_model": bank["model_name"],
            "prompt_pretrained": bank["pretrained"],
            "args": vars(args),
        },
        checkpoint_path,
    )
    results = {
        "class_names": bank["class_names"],
        "best_validation_seen_accuracy": best_seen_accuracy,
        "test": test_metrics,
        "test_diagnostics": test_diagnostics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)

    print("\nGeneralized open-vocabulary test")
    print("  overall: {:.2f}%".format(100.0 * test_metrics["overall_accuracy"]))
    print("  seen:    {:.2f}%".format(100.0 * test_metrics["seen_accuracy"]))
    print("  unseen:  {:.2f}%".format(100.0 * test_metrics["unseen_accuracy"]))
    print("  H:       {:.2f}%".format(100.0 * test_metrics["harmonic_mean"]))
    print("Saved: {}".format(checkpoint_path))


if __name__ == "__main__":
    main()

