"""Training entry point for the proposed hybrid model or a baseline."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import FocalLoss
from src.models.baselines import build_baseline
from src.models.hybrid_cnn_transformer import HybridCNNTransformer
from src.preprocessing.dataset import CLASS_NAMES, HAM10000Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, cfg: dict, num_classes: int) -> torch.nn.Module:
    if name == "hybrid_cnn_transformer":
        m = cfg["model"]
        return HybridCNNTransformer(
            num_classes=num_classes,
            cnn_backbone=m["cnn_backbone"],
            transformer_layers=m["transformer_layers"],
            transformer_heads=m["transformer_heads"],
            transformer_dim=m["transformer_dim"],
            mlp_ratio=m["mlp_ratio"],
            dropout=m["dropout"],
        )
    return build_baseline(name, num_classes=num_classes)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "val"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="hybrid_cnn_transformer",
                         choices=["hybrid_cnn_transformer", "resnet50", "efficientnet_b0", "vit_base"])
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, prep_cfg, train_cfg = cfg["data"], cfg["preprocessing"], cfg["training"]

    train_ds = HAM10000Dataset(
        split_csv=f"{data_cfg['splits_dir']}/train.csv", images_dir=data_cfg["images_dir"],
        image_size=data_cfg["image_size"], train=True,
        hair_removal=prep_cfg["hair_removal"], clahe=prep_cfg["clahe"],
    )
    val_ds = HAM10000Dataset(
        split_csv=f"{data_cfg['splits_dir']}/val.csv", images_dir=data_cfg["images_dir"],
        image_size=data_cfg["image_size"], train=False,
        hair_removal=prep_cfg["hair_removal"], clahe=prep_cfg["clahe"],
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                               num_workers=train_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                             num_workers=train_cfg["num_workers"])

    model = build_model(args.model, cfg, num_classes=len(CLASS_NAMES)).to(device)
    criterion = FocalLoss(alpha=train_cfg["focal_alpha"], gamma=train_cfg["focal_gamma"]) \
        if train_cfg["loss"] == "focal" else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    run_dir = Path(args.run_dir or f"runs/{args.model}")
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc, patience_counter = 0.0, 0
    for epoch in range(train_cfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"epoch {epoch}: train {train_metrics} val {val_metrics}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val accuracy: {best_val_acc:.4f}. Checkpoint saved to {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
