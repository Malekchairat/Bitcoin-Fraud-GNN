from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "outputs"
CLASSES_PATH = RAW_DIR / "elliptic_txs_classes.csv"
EDGES_PATH = RAW_DIR / "elliptic_txs_edgelist.csv"
FEATURES_PATH = RAW_DIR / "elliptic_txs_features.csv"

LABEL_NAMES = {
    "1": "illicit",
    "0": "licit",
    "-1": "unknown",
    "unknown": "unknown",
    "2": "unknown",
}


st.set_page_config(
    page_title="Elliptic Fraud Intelligence",
    page_icon="₿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827 0%, #172033 100%);
    }
    [data-testid="stSidebar"] * {color: #f8fafc;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 5px 18px rgba(15,23,42,.05);
    }
    [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #1e3a8a 100%);
        border-radius: 18px;
        color: white;
        padding: 28px 32px;
        margin-bottom: 22px;
    }
    .hero h1 {margin: 0 0 8px 0; color: white; font-size: 38px;}
    .hero p {margin: 0; color: #dbeafe; font-size: 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading transaction labels and time steps...")
def load_transactions() -> pd.DataFrame:
    processed = ROOT / "data" / "processed" / "dashboard_transactions.csv"
    if processed.exists():
        return pd.read_csv(processed)
    classes = pd.read_csv(CLASSES_PATH, dtype={"txId": "int64", "class": "string"})
    metadata = pd.read_csv(
        FEATURES_PATH,
        header=None,
        usecols=[0, 1],
        names=["txId", "time_step"],
        dtype={"txId": "int64", "time_step": "int64"},
    )
    transactions = metadata.merge(classes, on="txId", how="left", validate="one_to_one")
    transactions["label"] = transactions["class"].map(
        lambda value: LABEL_NAMES.get(str(value), "unknown")
    )
    return transactions


@st.cache_data(show_spinner="Loading graph relationships...")
def load_edges() -> pd.DataFrame:
    processed = ROOT / "data" / "processed" / "dashboard_edges.csv"
    if processed.exists():
        return pd.read_csv(processed)
    return pd.read_csv(EDGES_PATH, dtype="int64")


@st.cache_data
def feature_count() -> int:
    return len(pd.read_csv(FEATURES_PATH, header=None, nrows=1).columns) - 2


@st.cache_data
def load_model_results() -> pd.DataFrame:
    path = OUTPUTS_DIR / "tables" / "model_results.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"model", "f1", "precision", "recall", "roc_auc", "pr_auc"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame


def format_count(value: int) -> str:
    return f"{value:,}"


def show_metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def build_transaction_graph(
    transaction_id: int,
    edges: pd.DataFrame,
    transactions: pd.DataFrame,
    max_neighbors: int,
) -> tuple[str, pd.DataFrame] | None:
    connected = edges.loc[
        (edges["txId1"] == transaction_id) | (edges["txId2"] == transaction_id)
    ]
    if connected.empty:
        return None

    neighbors = pd.concat(
        [
            connected.loc[connected["txId1"] == transaction_id, "txId2"],
            connected.loc[connected["txId2"] == transaction_id, "txId1"],
        ]
    ).drop_duplicates()
    neighbors = neighbors.sort_values().head(max_neighbors)
    node_ids = {transaction_id, *neighbors.tolist()}
    visible_edges = connected[
        connected["txId1"].isin(node_ids) & connected["txId2"].isin(node_ids)
    ]
    labels = transactions.set_index("txId")["label"].to_dict()
    colors = {"illicit": "#e74c3c", "licit": "#2ecc71", "unknown": "#95a5a6"}

    dot_lines = [
        "graph transactions {",
        '  graph [overlap=false, bgcolor="transparent"];',
        '  node [style="filled", fontname="Arial", fontsize=10];',
        '  edge [color="#9aa4b2"];',
    ]
    for node_id in sorted(node_ids):
        label = labels.get(node_id, "unknown")
        color = colors[label]
        shape = "doublecircle" if node_id == transaction_id else "circle"
        dot_lines.append(
            f'  "{node_id}" [label="{node_id}\\n{label}", fillcolor="{color}", shape="{shape}"];'
        )
    for row in visible_edges.itertuples(index=False):
        dot_lines.append(f'  "{row.txId1}" -- "{row.txId2}";')
    dot_lines.append("}")

    visible_nodes = pd.DataFrame(
        {
            "txId": sorted(node_ids),
            "label": [labels.get(node_id, "unknown") for node_id in sorted(node_ids)],
        }
    )
    return "\n".join(dot_lines), visible_nodes


@st.cache_data(show_spinner="Preparing 3D graph coordinates...")
def prepare_3d_graph_data(
    transactions: pd.DataFrame,
    edges: pd.DataFrame,
    max_nodes: int,
    max_edges: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_nodes >= len(transactions):
        nodes = transactions.copy()
    else:
        nodes = transactions.sample(max_nodes, random_state=42).copy()

    transaction_ids = nodes["txId"].to_numpy(dtype="uint64")
    nodes["x"] = nodes["time_step"].astype(float)
    nodes["y"] = (
        (transaction_ids * np.uint64(2654435761)) % np.uint64(1000003)
    ).astype(float) / 1000003
    nodes["z"] = (
        (transaction_ids * np.uint64(2246822519)) % np.uint64(1000033)
    ).astype(float) / 1000033

    node_ids = set(nodes["txId"])
    visible_edges = edges[
        edges["txId1"].isin(node_ids) & edges["txId2"].isin(node_ids)
    ]
    if max_edges is not None and len(visible_edges) > max_edges:
        visible_edges = visible_edges.sample(max_edges, random_state=42)
    return nodes, visible_edges


def build_3d_graph_figure(nodes: pd.DataFrame, edges: pd.DataFrame) -> go.Figure:
    colors = {"illicit": "#e74c3c", "licit": "#2ecc71", "unknown": "#95a5a6"}
    figure = go.Figure()

    for label in ("unknown", "licit", "illicit"):
        subset = nodes[nodes["label"] == label]
        figure.add_trace(
            go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode="markers",
                name=label.title(),
                text=subset["txId"].astype(str),
                customdata=subset[["time_step"]],
                hovertemplate=(
                    "txId: %{text}<br>time step: %{customdata[0]}<br>"
                    f"label: {label}<extra></extra>"
                ),
                marker={
                    "size": 2.5 if len(nodes) > 50000 else 4,
                    "color": colors[label],
                    "opacity": 0.75,
                },
            )
        )

    if not edges.empty:
        coordinates = nodes.set_index("txId")[["x", "y", "z"]]
        edge_coordinates = (
            edges.join(coordinates, on="txId1")
            .rename(columns={"x": "x1", "y": "y1", "z": "z1"})
            .join(coordinates, on="txId2", rsuffix="_2")
            .rename(columns={"x": "x2", "y": "y2", "z": "z2"})
        )
        x_values = np.empty(len(edge_coordinates) * 3)
        y_values = np.empty(len(edge_coordinates) * 3)
        z_values = np.empty(len(edge_coordinates) * 3)
        x_values[::3] = edge_coordinates["x1"]
        x_values[1::3] = edge_coordinates["x2"]
        x_values[2::3] = np.nan
        y_values[::3] = edge_coordinates["y1"]
        y_values[1::3] = edge_coordinates["y2"]
        y_values[2::3] = np.nan
        z_values[::3] = edge_coordinates["z1"]
        z_values[1::3] = edge_coordinates["z2"]
        z_values[2::3] = np.nan
        figure.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                name="Relationships",
                line={"color": "#64748b", "width": 1},
                opacity=0.18,
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        height=760,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        scene={
            "xaxis_title": "Time step",
            "yaxis_title": "Layout coordinate",
            "zaxis_title": "Layout coordinate",
            "camera": {"projection": {"type": "perspective"}},
        },
        uirevision="bitcoin-graph",
    )
    return figure


if not CLASSES_PATH.exists() or not EDGES_PATH.exists() or not FEATURES_PATH.exists():
    st.error(
        "The raw Elliptic dataset is incomplete. Expected files in "
        f"`{RAW_DIR}`: classes, edgelist, and features CSV files."
    )
    st.stop()

with st.sidebar:
    st.markdown("## Fraud Intelligence")
    st.caption("Elliptic Bitcoin transaction explorer")
    st.markdown(
        "**Sections**\n"
        "1. Dataset overview\n"
        "2. Transaction lookup\n"
        "3. Local graph\n"
        "4. 3D overview\n"
        "5. Model comparison"
    )
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown(
        "**Quick guide**\n"
        "- **Illicit** is the positive class.\n"
        "- **Unknown** labels are excluded from supervised evaluation.\n"
        "- The notebook uses time steps 1–29 for training, 30–34 for validation, "
        "and 35–49 for testing."
    )

transactions = load_transactions()
edges = load_edges()
model_results = load_model_results()

st.markdown(
    """
    <div class="hero">
      <h1>Bitcoin Fraud Intelligence</h1>
      <p>Elliptic transaction graph · Temporal validation · GNN model comparison</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_transactions = len(transactions)
labelled = int((transactions["label"] != "unknown").sum())
illicit = int((transactions["label"] == "illicit").sum())

metrics = st.columns(5)
with metrics[0]:
    show_metric("Transactions", format_count(total_transactions))
with metrics[1]:
    show_metric("Graph edges", format_count(len(edges)))
with metrics[2]:
    show_metric("Features / node", format_count(feature_count()))
with metrics[3]:
    show_metric("Labelled", format_count(labelled))
with metrics[4]:
    show_metric("Illicit", format_count(illicit), "Positive class used by the notebook")

st.subheader("Dataset overview")
left, right = st.columns(2)

with left:
    label_summary = (
        transactions["label"]
        .value_counts()
        .rename_axis("label")
        .rename("transactions")
        .reindex(["licit", "illicit", "unknown"])
        .fillna(0)
        .astype(int)
    )
    st.markdown("#### Transaction labels")
    st.bar_chart(label_summary)
    illicit_rate = illicit / labelled if labelled else 0
    st.caption(
        f"Illicit rate among labelled transactions: **{illicit_rate:.2%}**. "
        "The unknown majority is not treated as licit."
    )

with right:
    split = pd.cut(
        transactions["time_step"],
        bins=[0, 29, 34, 49],
        labels=["Train (1–29)", "Validation (30–34)", "Test (35–49)"],
    )
    split_summary = (
        transactions.assign(split=split)
        .groupby(["split", "label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["licit", "illicit", "unknown"], fill_value=0)
    )
    st.markdown("#### Notebook temporal split")
    st.dataframe(split_summary, width="stretch")
    st.caption(
        "The split is chronological, so validation and test transactions occur "
        "after the training period."
    )

st.subheader("Transaction activity over time")
temporal = (
    transactions.groupby("time_step")
    .agg(
        transactions=("txId", "size"),
        labelled=("label", lambda values: int((values != "unknown").sum())),
        illicit=("label", lambda values: int((values == "illicit").sum())),
    )
)
temporal["illicit_rate_labelled"] = (
    temporal["illicit"] / temporal["labelled"].replace(0, pd.NA)
).astype("float64")
tab_activity, tab_rate = st.tabs(["Transaction volume", "Illicit rate"])
with tab_activity:
    st.line_chart(temporal[["transactions", "labelled", "illicit"]])
with tab_rate:
    st.line_chart(temporal[["illicit_rate_labelled"]])

st.subheader("Look up a transaction")
query = st.text_input("Transaction ID", placeholder="For example: 230425980")
if query:
    try:
        transaction_id = int(query)
    except ValueError:
        st.warning("Enter a numeric transaction ID.")
    else:
        match = transactions[transactions["txId"] == transaction_id]
        if match.empty:
            st.info("That transaction ID is not present in the dataset.")
        else:
            record = match.iloc[0]
            st.dataframe(
                pd.DataFrame(
                    {
                        "Field": ["Transaction ID", "Time step", "Label"],
                        "Value": [
                            str(format_count(int(record["txId"]))),
                            str(int(record["time_step"])),
                            str(record["label"]),
                        ],
                    }
                ),
                hide_index=True,
                width="stretch",
            )

st.subheader("Transaction graph")
st.caption(
    "Select a transaction to see its direct graph neighborhood. "
    "The center node is outlined; colors represent the known class."
)
graph_query = st.text_input(
    "Center transaction ID",
    value=query,
    placeholder="Enter a transaction ID",
    key="graph_transaction_id",
)
neighbor_limit = st.slider("Maximum neighbors to display", 5, 50, 20)
if graph_query:
    try:
        graph_transaction_id = int(graph_query)
    except ValueError:
        st.warning("Enter a numeric transaction ID for the graph.")
    else:
        if graph_transaction_id not in set(transactions["txId"]):
            st.info("That transaction ID is not present in the dataset.")
        else:
            graph_result = build_transaction_graph(
                graph_transaction_id, edges, transactions, neighbor_limit
            )
            if graph_result is None:
                st.info("This transaction has no listed graph relationships.")
            else:
                graph_dot, graph_nodes = graph_result
                st.graphviz_chart(graph_dot)
                st.caption(
                    f"Showing {len(graph_nodes) - 1} neighbor(s) and "
                    f"{len(graph_nodes)} node(s) total. "
                    "Unknown labels are shown in gray."
                )
                with st.expander("Show displayed nodes"):
                    st.dataframe(graph_nodes, hide_index=True, width="stretch")
else:
    st.info("Enter a transaction ID above to render its graph neighborhood.")

st.subheader("3D graph overview")
st.markdown(
    "Rotate by dragging, zoom with the wheel, and pan with shift-drag. "
    "The x-axis is the transaction time step; the other two axes provide a "
    "stable visual layout so the full network can be explored in WebGL."
)
full_graph = st.checkbox(
    "Use all 203,769 transaction nodes",
    value=False,
    help="The full node set can use substantial browser memory.",
)
if full_graph:
    graph_node_limit = len(transactions)
    graph_edge_limit: int | None = None
    st.warning(
        "Full-node mode can be slow because the browser must render more than "
        "200,000 points. Disable it for a faster interactive overview."
    )
else:
    graph_node_limit = st.slider(
        "Nodes in interactive overview", 1_000, 50_000, 12_000, step=1_000
    )
    graph_edge_limit = st.slider(
        "Relationships in interactive overview", 1_000, 100_000, 25_000, step=1_000
    )

if st.button("Render 3D graph", type="primary"):
    graph_nodes, graph_edges = prepare_3d_graph_data(
        transactions, edges, graph_node_limit, graph_edge_limit
    )
    st.session_state["graph_3d_nodes"] = graph_nodes
    st.session_state["graph_3d_edges"] = graph_edges

if "graph_3d_nodes" in st.session_state:
    graph_nodes = st.session_state["graph_3d_nodes"]
    graph_edges = st.session_state["graph_3d_edges"]
    st.plotly_chart(
        build_3d_graph_figure(graph_nodes, graph_edges),
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.caption(
        f"Displayed {len(graph_nodes):,} nodes and {len(graph_edges):,} "
        "relationships. The full graph uses all available relationships; "
        "sampled mode limits both for browser performance."
    )

st.subheader("Model results")
if model_results.empty:
    st.info(
        "No model comparison table has been written to `outputs/tables` yet. "
        "Run `python train_models.py --normal` and this section will "
        "automatically display the saved result table."
    )
else:
    metric_columns = ["f1", "precision", "recall", "roc_auc", "pr_auc"]
    seed_count = int(model_results["seed"].nunique()) if "seed" in model_results else 1
    st.caption(
        f"Final test results across {seed_count} seed(s). "
        "Values are mean ± standard deviation."
    )
    aggregate = (
        model_results.groupby("model", as_index=False)[metric_columns]
        .agg(["mean", "std"])
    )
    aggregate.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    display_columns = ["model"] + [
        f"{metric}_mean"
        for metric in metric_columns
    ]
    aggregate = aggregate[display_columns].rename(
        columns={f"{metric}_mean": metric for metric in metric_columns}
    )
    st.dataframe(
        aggregate.style.format(
            {metric: "{:.4f}" for metric in metric_columns}
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("View seed-by-seed results"):
        st.dataframe(model_results, hide_index=True, width="stretch")

st.caption(
    "This dashboard summarizes data and saved results; it does not retrain the "
    "GNN on every page refresh."
)
