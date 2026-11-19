import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data

from elliptic_dataset import load_elliptic_dataset
from models import GAT, GraphSAGE


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def temporal_masks(labels, timestamps, train_end=34, validation_end=38):
    known = labels >= 0
    train_mask = known & (timestamps < train_end)
    val_mask = known & (timestamps >= train_end) & (timestamps < validation_end)
    test_mask = known & (timestamps >= validation_end)
    return train_mask, val_mask, test_mask


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = logits.softmax(dim=-1)[:, 1].cpu().numpy()
        predictions = logits.argmax(dim=-1).cpu().numpy()
        labels = data.y.cpu().numpy()
        selected = mask.cpu().numpy()
    y_true = labels[selected]
    y_score = probabilities[selected]
    y_pred = predictions[selected]
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        average_precision = float("nan")
    else:
        auc = roc_auc_score(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "average_precision": average_precision,
        "predictions": y_pred,
        "probabilities": y_score,
    }


def train_model(model, data, train_mask, val_mask, epochs=100, learning_rate=0.005, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    history = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(val_logits[val_mask], data.y[val_mask]).item()
        history.append({"epoch": epoch + 1, "loss": loss.item(), "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def create_model(model_name, in_channels):
    if model_name.lower() in ("graphsage", "graph_sage", "sage"):
        return GraphSAGE(in_channels)
    if model_name.lower() in ("gat", "graph_attention"):
        return GAT(in_channels)
    raise ValueError("model_name must be 'GraphSAGE' or 'GAT'")


def main():
    parser = argparse.ArgumentParser(description="Train a GNN on the Elliptic dataset")
    parser.add_argument("data_dir")
    parser.add_argument("--model", choices=["GraphSAGE", "GAT"], default="GraphSAGE")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="model.pt")
    args = parser.parse_args()

    set_seed()
    x, edge_index, y, tx_ids, timestamps = load_elliptic_dataset(args.data_dir)
    data = Data(x=x, edge_index=edge_index, y=y)
    train_mask, val_mask, test_mask = temporal_masks(y, timestamps)
    model = create_model(args.model, x.shape[1])
    model, history = train_model(model, data, train_mask, val_mask, epochs=args.epochs)

    data_cpu = data.cpu()
    model_cpu = model.cpu()
    metrics = evaluate(model_cpu, data_cpu, test_mask)
    print({key: value for key, value in metrics.items() if key not in ("predictions", "probabilities")})
    torch.save(
        {
            "state_dict": model_cpu.state_dict(),
            "model_name": args.model,
            "in_channels": x.shape[1],
            "history": history,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
