"""Generate and load an auditable CLIP text prompt bank.

The resulting ``.pt`` file is consumed by ``OVDGMamba``. Only the frozen CLIP
text tower is used; HSI and DSM/LiDAR data never pass through CLIP's image
encoder.
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F


DEFAULT_TEMPLATES = [
    "an aerial remote sensing image of {label}",
    "a hyperspectral remote sensing observation of {label}",
    "a land-cover patch containing {label}",
    "a top-down view of {label}",
    "{description}",
]


def load_prompt_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    classes = config.get("classes", [])
    if not classes:
        raise ValueError("Prompt config must define a non-empty 'classes' list")
    names = [item["name"] for item in classes]
    if len(names) != len(set(names)):
        raise ValueError("Class names in the prompt config must be unique")
    return config


def build_class_prompts(class_record: Dict, templates: List[str]) -> List[str]:
    """Create a de-duplicated prompt ensemble for one class."""
    label = class_record["name"].replace("_", " ")
    description = class_record.get("description", label).strip()
    aliases = [alias.strip() for alias in class_record.get("aliases", [])]
    labels = [label] + [alias for alias in aliases if alias]
    prompts = []
    for candidate in labels:
        for template in templates:
            prompt = template.format(label=candidate, description=description).strip()
            if prompt and prompt not in prompts:
                prompts.append(prompt)
    return prompts


def _encode_prompts(model, tokenizer, prompts, device, batch_size):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            tokens = tokenizer(batch_prompts).to(device)
            features = model.encode_text(tokens)
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def generate_prompt_bank(
    config_path: str,
    output_path: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "auto",
    batch_size: int = 64,
) -> Dict:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "open_clip is required only to generate the prompt bank. "
            "Install it with: pip install open_clip_torch"
        ) from exc

    config = load_prompt_config(config_path)
    templates = config.get("templates", DEFAULT_TEMPLATES)
    # The current relational head expects seen classes first. Sorting is stable,
    # and the saved class_to_index mapping is the authoritative label order.
    classes = sorted(config["classes"], key=lambda item: not item.get("seen", False))
    num_seen_classes = sum(bool(item.get("seen", False)) for item in classes)
    if num_seen_classes == 0:
        raise ValueError("At least one class must have seen=true")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device_obj).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    class_embeddings = []
    prompt_manifest = {}
    for class_record in classes:
        prompts = build_class_prompts(class_record, templates)
        prompt_features = _encode_prompts(
            model, tokenizer, prompts, device_obj, batch_size
        )
        embedding = F.normalize(prompt_features.mean(dim=0), dim=-1)
        class_embeddings.append(embedding)
        prompt_manifest[class_record["name"]] = prompts

    class_names = [item["name"] for item in classes]
    bank = {
        "schema_version": 1,
        "embeddings": torch.stack(class_embeddings, dim=0),
        "class_names": class_names,
        "class_to_index": {name: index for index, name in enumerate(class_names)},
        "seen_mask": torch.tensor(
            [bool(item.get("seen", False)) for item in classes], dtype=torch.bool
        ),
        "num_seen_classes": num_seen_classes,
        "prompts": prompt_manifest,
        "model_name": model_name,
        "pretrained": pretrained,
        "source_config": os.path.abspath(config_path),
    }
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(bank, output_path)
    return bank


def load_prompt_bank(path: str, map_location: str = "cpu") -> Dict:
    bank = torch.load(path, map_location=map_location)
    required = {
        "embeddings", "class_names", "class_to_index", "seen_mask",
        "num_seen_classes", "prompts", "model_name", "pretrained",
    }
    missing = required.difference(bank)
    if missing:
        raise ValueError("Prompt bank is missing fields: {}".format(sorted(missing)))
    embeddings = bank["embeddings"]
    if embeddings.dim() != 2 or embeddings.shape[0] != len(bank["class_names"]):
        raise ValueError("Prompt-bank embedding shape does not match class names")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Prompt bank contains non-finite embeddings")
    return bank


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a CLIP text prompt bank")
    parser.add_argument("--config", required=True, help="Prompt JSON configuration")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained", default="openai", help="Pretrained tag or local checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    bank = generate_prompt_bank(
        config_path=args.config,
        output_path=args.output,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )
    print("Prompt bank saved to: {}".format(args.output))
    print("Text model: {} ({})".format(bank["model_name"], bank["pretrained"]))
    print("Classes: {} (seen: {})".format(
        len(bank["class_names"]), bank["num_seen_classes"]
    ))
    print("Embedding shape: {}".format(tuple(bank["embeddings"].shape)))
    print("Class order: {}".format(", ".join(bank["class_names"])))


if __name__ == "__main__":
    main()

