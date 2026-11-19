"""Export verified notebook/model tables to the static React interface."""

import json
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from train_models import GAT, GraphSAGE, MLP, load_graph


ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "outputs" / "tables"
PUBLIC_DIR = ROOT / "web" / "public"


def layout_value(tx_id: int, salt: int) -> float:
    value = (int(tx_id) * salt) % 1000003
    return value / 1000003


def export_graph_data() -> None:
    transactions = pd.read_csv(
        ROOT / "data" / "processed" / "dashboard_transactions.csv"
    )
    edges = pd.read_csv(ROOT / "data" / "processed" / "dashboard_edges.csv")

    frame = transactions.sort_values(["time_step", "txId"])
    illicit = frame[frame["label"] == "illicit"].sample(
        n=min(300, int((frame["label"] == "illicit").sum())),
        random_state=42,
    )
    licit = frame[frame["label"] == "licit"].sample(
        n=min(300, int((frame["label"] == "licit").sum())),
        random_state=42,
    )
    unknown = frame[frame["label"] == "unknown"].sample(
        n=min(400, int((frame["label"] == "unknown").sum())),
        random_state=42,
    )
    seeds = pd.concat([illicit, licit, unknown])

    adjacency = {}
    for row in edges.itertuples(index=False):
        source = int(row.txId1)
        target = int(row.txId2)
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, []).append(source)

    seed_ids = seeds["txId"].astype(int).tolist()
    selected_ids = set(seed_ids)
    queue = deque(seed_ids)
    while queue and len(selected_ids) < 2000:
        current = queue.popleft()
        for neighbor_id in adjacency.get(current, []):
            neighbor_id = int(neighbor_id)
            if neighbor_id not in selected_ids:
                selected_ids.add(neighbor_id)
                queue.append(neighbor_id)
                if len(selected_ids) >= 2000:
                    break

    sampled = frame[frame["txId"].isin(selected_ids)].sort_values(
        ["time_step", "txId"]
    )

    nodes = []
    for row in sampled.itertuples(index=False):
        tx_id = int(row.txId)
        time_step = int(row.time_step)
        nodes.append(
            {
                "txId": tx_id,
                "timeStep": time_step,
                "label": str(row.label),
                "x": 35 + ((time_step - 1) / 48) * 930,
                "y": 30 + layout_value(tx_id, 2654435761) * 590,
            }
        )

    node_ids = {node["txId"] for node in nodes}
    visible_edges = edges[
        edges["txId1"].isin(node_ids) & edges["txId2"].isin(node_ids)
    ]
    if len(visible_edges) > 6000:
        visible_edges = visible_edges.sample(6000, random_state=42)

    graph = {
        "nodes": nodes,
        "edges": [
            {"source": int(row.txId1), "target": int(row.txId2)}
            for row in visible_edges.itertuples(index=False)
        ],
    }
    (PUBLIC_DIR / "graph-data.json").write_text(
        json.dumps(graph, indent=2),
        encoding="utf-8",
    )
    print(
        f"Exported graph data: {len(nodes)} nodes and {len(graph['edges'])} edges"
    )


def export_model_predictions(results: pd.DataFrame) -> None:
    graph_path = PUBLIC_DIR / "graph-data.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    unknown_nodes = [
        node["txId"] for node in graph["nodes"] if node["label"] == "unknown"
    ]
    if not unknown_nodes:
        (PUBLIC_DIR / "model-predictions.json").write_text("{}", encoding="utf-8")
        return

    data, frame, _, _, _ = load_graph()
    id_to_index = pd.Series(
        np.arange(len(frame)), index=frame["txId"].to_numpy()
    ).to_dict()
    indices = np.asarray(
        [id_to_index[tx_id] for tx_id in unknown_nodes], dtype=np.int64
    )

    model_factories = {
        "MLP": lambda: MLP(data.x.shape[1]),
        "GraphSAGE": lambda: GraphSAGE(data.x.shape[1]),
        "GAT": lambda: GAT(data.x.shape[1]),
    }
    predictions = {str(tx_id): {} for tx_id in unknown_nodes}
    device = torch.device("cpu")
    data = data.to(device)

    for name, factory in model_factories.items():
        seed_rows = results[(results["model"] == name) & (results["seed"] == 46)]
        if seed_rows.empty:
            continue
        seed_row = seed_rows.iloc[-1]
        threshold = float(seed_row["threshold"])
        model = factory().to(device)
        model.load_state_dict(
            torch.load(
                ROOT / "outputs" / "models" / f"{name.lower()}_seed_46.pt",
                map_location=device,
                weights_only=True,
            )
        )
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()

        for tx_id, probability in zip(unknown_nodes, probabilities[indices]):
            predictions[str(tx_id)][name] = {
                "label": "illicit" if float(probability) >= threshold else "licit",
                "probability": float(probability),
                "threshold": threshold,
            }

    (PUBLIC_DIR / "model-predictions.json").write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    print(
        f"Exported predictions for {len(unknown_nodes)} unknown graph nodes"
    )


def main() -> None:
    results_path = TABLE_DIR / "model_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No model results found at {results_path}. Run the normal experiment first."
        )

    results = pd.read_csv(results_path)
    required = {"model", "seed", "f1", "precision", "recall", "roc_auc", "pr_auc"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"model_results.csv is missing columns: {sorted(missing)}")

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DIR / "model-results.json").write_text(
        json.dumps(results.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    metadata = {
        "source": str(results_path),
        "seed_count": int(results["seed"].nunique()),
        "seeds": sorted(int(seed) for seed in results["seed"].unique()),
        "model_count": int(results["model"].nunique()),
        "row_count": len(results),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "mode": "normal" if int(results["seed"].nunique()) >= 5 else "non-normal",
    }
    (PUBLIC_DIR / "model-results-meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    export_graph_data()
    export_model_predictions(results)
    print(
        f"Exported {len(results)} real result rows "
        f"from {results_path} to {PUBLIC_DIR}"
    )


if __name__ == "__main__":
    main()
