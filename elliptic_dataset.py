from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


def load_elliptic_dataset(
    data_dir,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Elliptic CSV files and return features, edge index, labels, ids, and timestamps."""
    data_dir = Path(data_dir)

    feature_path = data_dir / "elliptic_txs_features.csv"
    class_path = data_dir / "elliptic_txs_classes.csv"
    edge_path = data_dir / "elliptic_txs_edgelist.csv"

    raw_features = pd.read_csv(feature_path)
    if "txId" in raw_features.columns:
        features = raw_features
    else:
        features = pd.read_csv(feature_path, header=None)
        features.columns = ["txId", "time"] + [
            f"feature_{i}" for i in range(features.shape[1] - 2)
        ]

    raw_classes = pd.read_csv(class_path)
    if "txId" in raw_classes.columns:
        classes = raw_classes
    else:
        classes = pd.read_csv(class_path, header=None)
        classes.columns = ["txId", "class"]

    raw_edges = pd.read_csv(edge_path)
    if "txId1" in raw_edges.columns and "txId2" in raw_edges.columns:
        edges = raw_edges
    else:
        edges = pd.read_csv(edge_path, header=None)
        edges.columns = ["txId1", "txId2"]
    tx_ids = features["txId"].astype(str)
    id_to_index = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    feature_columns = [
        column
        for column in features.columns
        if column not in ("txId", "time", "Time step", "time_step")
    ]
    x = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(
        dtype=np.float32
    )

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / np.where(std > 1e-8, std, 1.0)

    class_map = dict(
        zip(classes.iloc[:, 0].astype(str), classes.iloc[:, 1].astype(str))
    )
    y_raw = np.array([class_map.get(tx_id, "unknown") for tx_id in tx_ids])
    y = np.full(len(tx_ids), -1, dtype=np.int64)
    y[y_raw == "1"] = 1
    y[y_raw == "2"] = 0

    timestamp_column = next(
        (column for column in ("time", "Time step", "time_step") if column in features),
        None,
    )
    if timestamp_column is not None:
        timestamps = features[timestamp_column].to_numpy(dtype=np.int64)
    else:
        timestamp = pd.to_numeric(features.iloc[:, 1], errors="coerce").fillna(0)
        timestamps = timestamp.to_numpy(dtype=np.int64)

    first_col = edges.columns[0]
    second_col = edges.columns[1]
    source = edges[first_col].astype(str).map(id_to_index)
    target = edges[second_col].astype(str).map(id_to_index)
    valid = source.notna() & target.notna()
    edge_rows = torch.tensor(
        np.column_stack([source[valid].to_numpy(), target[valid].to_numpy()]),
        dtype=torch.long,
    ).t().contiguous()

    if edge_rows.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.cat([edge_rows, edge_rows.flip(0)], dim=1)

    return (
        torch.from_numpy(x),
        edge_index,
        torch.from_numpy(y),
        torch.arange(len(tx_ids), dtype=torch.long),
        torch.from_numpy(timestamps.copy()),
    )
