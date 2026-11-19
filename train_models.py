"""Train the MLP, GraphSAGE, and GAT transaction classifiers."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import to_undirected


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"
TRAIN_STEPS = range(1, 30)
VAL_STEPS = range(30, 35)
TEST_STEPS = range(35, 50)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def best_threshold_on_val(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    thresholds = np.arange(0.05, 0.951, 0.01)
    scores = np.asarray(
        [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    )
    return float(thresholds[int(np.argmax(scores))]), thresholds, scores


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return self.network(x)


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.conv1(x, edge_index)))
        x = self.dropout(torch.relu(self.conv2(x, edge_index)))
        return self.output(x)


class GAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2, dropout=0.15)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=False, dropout=0.15)
        self.output = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.conv1(x, edge_index)))
        x = self.dropout(torch.relu(self.conv2(x, edge_index)))
        return self.output(x)


def load_graph() -> tuple[Data, pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None)
    features.columns = (
        ["txId", "time_step"]
        + [f"local_{i}" for i in range(1, 94)]
        + [f"agg_{i}" for i in range(1, 73)]
    )
    classes = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv")
    edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
    frame = features[["txId", "time_step"]].merge(classes, on="txId", how="left")
    frame["label"] = frame["class"].astype(str).map({"1": 1, "2": 0, "unknown": -1})
    if frame["label"].isna().any():
        raise ValueError("The class table contains unmapped values.")

    feature_columns = [f"local_{i}" for i in range(1, 94)] + [
        f"agg_{i}" for i in range(1, 73)
    ]
    train_nodes = frame.time_step.isin(TRAIN_STEPS).to_numpy()
    scaler = StandardScaler().fit(features.loc[train_nodes, feature_columns])
    x_scaled = scaler.transform(features[feature_columns]).astype(np.float32)

    id_to_idx = pd.Series(np.arange(len(frame)), index=frame.txId).to_dict()
    source_idx = edges.iloc[:, 0].map(id_to_idx).to_numpy()
    target_idx = edges.iloc[:, 1].map(id_to_idx).to_numpy()
    edge_index = to_undirected(
        torch.tensor(np.vstack([source_idx, target_idx]), dtype=torch.long)
    )
    labels = frame.label.to_numpy(dtype=np.int64)
    data = Data(
        x=torch.tensor(x_scaled),
        edge_index=edge_index,
        y=torch.tensor(labels),
        time_step=torch.tensor(frame.time_step.to_numpy(dtype=np.int64)),
    )
    train_mask = torch.tensor(train_nodes & (labels != -1))
    val_mask = torch.tensor(
        frame.time_step.isin(VAL_STEPS).to_numpy() & (labels != -1)
    )
    test_mask = torch.tensor(
        frame.time_step.isin(TEST_STEPS).to_numpy() & (labels != -1)
    )
    return data, frame, train_mask, val_mask, test_mask


def train_one(
    name: str,
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int,
    patience: int,
    class_weights: torch.Tensor,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    stalled_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
        val_y = data.y[val_mask].numpy()
        val_probabilities = probabilities[val_mask].numpy()
        threshold, _, _ = best_threshold_on_val(val_y, val_probabilities)
        val_metrics = evaluate(val_y, val_probabilities, threshold)
        history.append(
            {
                "model": name,
                "epoch": epoch,
                "loss": float(loss.item()),
                "val_threshold": threshold,
                "val_f1": val_metrics["f1"],
                "val_pr_auc": val_metrics["pr_auc"],
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    val_y = data.y[val_mask].numpy()
    val_probabilities = probabilities[val_mask].numpy()
    threshold, thresholds, threshold_scores = best_threshold_on_val(
        val_y, val_probabilities
    )
    test_y = data.y[test_mask].numpy()
    test_probabilities = probabilities[test_mask].numpy()
    test_metrics = evaluate(test_y, test_probabilities, threshold)
    result = {
        "model": name,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_f1": float(best_val_f1),
        "threshold": test_metrics.pop("threshold"),
        **test_metrics,
    }
    threshold_table = {
        "model": name,
        "threshold": thresholds,
        "validation_f1": threshold_scores,
    }
    return result, history, threshold_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normal",
        action="store_true",
        help="Run the notebook-normal experiment with five seeds and up to 200 epochs.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Run the lower-cost experiment with three seeds and up to 60 epochs.",
    )
    args = parser.parse_args(argv)
    if args.balanced and args.normal:
        parser.error("--normal and --balanced cannot be used together")
    fast_mode = not args.normal and not args.balanced
    epochs = 30 if fast_mode else (200 if args.normal else 60)
    patience = 5 if fast_mode else (30 if args.normal else 10)
    seeds = [42] if fast_mode else ([42, 43, 44, 45, 46] if args.normal else [42, 43, 44])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data, frame, train_mask, val_mask, test_mask = load_graph()
    data = data.to(device)
    n_licit = int(((data.y == 0) & train_mask.to(device)).sum())
    n_illicit = int(((data.y == 1) & train_mask.to(device)).sum())
    class_weights = torch.tensor(
        [1.0, n_licit / n_illicit], dtype=torch.float32, device=device
    )

    all_results = []
    all_history = []
    all_thresholds = []
    for seed in seeds:
        set_seed(seed)
        models = {
            "MLP": MLP(data.x.shape[1]),
            "GraphSAGE": GraphSAGE(data.x.shape[1]),
            "GAT": GAT(data.x.shape[1]),
        }
        for name, model in models.items():
            model = model.to(device)
            result, history, threshold_table = train_one(
                name,
                model,
                data,
                train_mask.to(device),
                val_mask.to(device),
                test_mask.to(device),
                epochs,
                patience,
                class_weights,
            )
            result.update(
                {
                    "seed": seed,
                    "device": str(device),
                    "fast_mode": fast_mode,
                    "train_nodes": int(train_mask.sum()),
                    "validation_nodes": int(val_mask.sum()),
                    "test_nodes": int(test_mask.sum()),
                }
            )
            all_results.append(result)
            all_history.extend(history)
            all_thresholds.append(
                pd.DataFrame(
                    {
                        "model": threshold_table["model"],
                        "threshold": threshold_table["threshold"],
                        "validation_f1": threshold_table["validation_f1"],
                        "seed": seed,
                    }
                )
            )
            torch.save(model.state_dict(), MODEL_DIR / f"{name.lower()}_seed_{seed}.pt")
            print(
                f"{name}: test_f1={result['f1']:.4f}, "
                f"test_pr_auc={result['pr_auc']:.4f}, "
                f"threshold={result['threshold']:.2f}"
            )

    pd.DataFrame(all_results).to_csv(TABLE_DIR / "model_results.csv", index=False)
    pd.DataFrame(all_history).to_csv(TABLE_DIR / "training_history.csv", index=False)
    pd.concat(all_thresholds, ignore_index=True).to_csv(
        TABLE_DIR / "threshold_search.csv", index=False
    )
    print(f"Saved results to {TABLE_DIR}")
    print(f"Dataset: {len(frame):,} nodes, {data.edge_index.shape[1]:,} graph edges")


if __name__ == "__main__":
    main()
