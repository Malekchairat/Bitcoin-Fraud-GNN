import tempfile
from pathlib import Path

import pandas as pd
import torch

from elliptic_dataset import load_elliptic_dataset
from models import GAT, GraphSAGE
from train import temporal_masks, create_model


def write_elliptic_csvs(directory):
    pd.DataFrame(
        [
            [1, 1, 0.1, 0.2],
            [2, 1, 0.3, 0.4],
            [3, 2, 0.5, 0.6],
            [4, 2, 0.7, 0.8],
        ],
        columns=["txId", "time", "feature_1", "feature_2"],
    ).to_csv(directory / "elliptic_txs_features.csv", index=False)
    pd.DataFrame(
        [[1, "1"], [2, "2"], [3, "2"], [4, "unknown"]],
        columns=["txId", "class"],
    ).to_csv(directory / "elliptic_txs_classes.csv", index=False)
    pd.DataFrame(
        [[1, 2], [2, 3], [3, 4]],
        columns=["txId1", "txId2"],
    ).to_csv(directory / "elliptic_txs_edgelist.csv", index=False)


def test_load_elliptic_dataset():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        write_elliptic_csvs(directory)
        x, edge_index, y, tx_ids, timestamps = load_elliptic_dataset(directory)

    assert x.shape == (4, 2)
    assert edge_index.shape == (2, 6)
    assert y.tolist() == [1, 0, 0, -1]
    assert tx_ids.tolist() == [0, 1, 2, 3]
    assert timestamps.tolist() == [1, 1, 2, 2]


def test_temporal_masks():
    labels = torch.tensor([0, 1, -1, 1])
    timestamps = torch.tensor([10, 35, 40, 45])
    train_mask, val_mask, test_mask = temporal_masks(labels, timestamps)
    assert train_mask.tolist() == [True, False, False, False]
    assert val_mask.tolist() == [False, True, False, False]
    assert test_mask.tolist() == [False, False, False, True]


def test_model_factories_and_forward_shapes():
    x = torch.randn(4, 5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    for model in (create_model("GraphSAGE", 5), create_model("GAT", 5)):
        assert model(x, edge_index).shape == (4, 2)
