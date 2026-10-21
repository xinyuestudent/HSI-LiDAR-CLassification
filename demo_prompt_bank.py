"""Load a generated prompt bank and connect it to OVDG-Mamba."""

import argparse

import torch

from clip_prompt_bank import load_prompt_bank
from ovdg_model import create_ovdg_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--bands", type=int, default=63)
    parser.add_argument("--patch-size", type=int, default=9)
    args = parser.parse_args()

    bank = load_prompt_bank(args.bank)
    model = create_ovdg_model(
        text_embeddings=bank["embeddings"],
        num_seen_classes=bank["num_seen_classes"],
        hsi_embed_dim=32,
        feature_dim=32,
        aux_channels=1,
        d_state=8,
        expand=1,
    )
    model.eval()

    hsi = torch.randn(2, args.patch_size, args.patch_size, args.bands)
    dsm = torch.randn(2, 1, args.patch_size, args.patch_size)
    # Replace this debugging grid with real sensor centre wavelengths in formal runs.
    wavelengths = torch.linspace(400.0, 1000.0, args.bands)
    with torch.no_grad():
        outputs = model(hsi, dsm, wavelengths)
        prediction = outputs["logits"].argmax(dim=-1)

    print("Bank model: {} ({})".format(bank["model_name"], bank["pretrained"]))
    print("Vocabulary: {}".format(bank["class_names"]))
    print("Logits shape: {}".format(tuple(outputs["logits"].shape)))
    print("Predicted labels: {}".format(
        [bank["class_names"][index] for index in prediction.tolist()]
    ))


if __name__ == "__main__":
    main()

