import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
from sklearn.decomposition import PCA

from elliptic_dataset import load_elliptic_dataset
from models import GAT, GraphSAGE
from train import evaluate, set_seed, temporal_masks, train_model
from torch_geometric.data import Data


st.set_page_config(page_title="Bitcoin Fraud Detection", layout="wide")
st.title("Bitcoin Fraud Detection with Graph Neural Networks")


@st.cache_data
def load_data(data_dir):
    x, edge_index, y, tx_ids, timestamps = load_elliptic_dataset(data_dir)
    return x, edge_index, y, tx_ids, timestamps


data_dir = st.sidebar.text_input(
    "Elliptic data directory",
    value="data",
    help="Directory containing elliptic_txs_features.csv, elliptic_txs_classes.csv, and elliptic_txs_edgelist.csv",
)

model_name = st.sidebar.selectbox("Model", ["GraphSAGE", "GAT"])
epochs = st.sidebar.slider("Training epochs", 10, 300, 100, 10)
run_training = st.sidebar.button("Train model")

try:
    x, edge_index, y, tx_ids, timestamps = load_data(data_dir)
except (FileNotFoundError, OSError, pd.errors.EmptyDataError) as exc:
    st.info("Set the Elliptic CSV directory to begin.")
    st.caption(str(exc))
    st.stop()

data = Data(x=x, edge_index=edge_index, y=y)
train_mask, val_mask, test_mask = temporal_masks(y, timestamps)

st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", len(x))
col2.metric("Edges", edge_index.shape[1] // 2)
col3.metric("Licit", int((y == 0).sum()))
col4.metric("Illicit", int((y == 1).sum()))

if run_training:
    set_seed()
    model = GraphSAGE(x.shape[1]) if model_name == "GraphSAGE" else GAT(x.shape[1])
    with st.spinner(f"Training {model_name}..."):
        model, history = train_model(
            model,
            data,
            train_mask,
            val_mask,
            epochs=epochs,
        )
    st.session_state["model"] = model
    st.session_state["history"] = history
    st.success("Training complete")

model = st.session_state.get("model")
if model is not None:
    metrics = evaluate(model, data, test_mask)
    st.subheader("Test evaluation")
    metric_columns = st.columns(4)
    for column, name in zip(metric_columns, ("accuracy", "f1", "auc", "average_precision")):
        value = metrics[name]
        column.metric(name.replace("_", " ").upper(), "N/A" if np.isnan(value) else f"{value:.4f}")

    history = st.session_state.get("history", [])
    if history:
        history_df = pd.DataFrame(history)
        st.subplot = getattr(st, "subplot", None)
        st.line_chart(history_df.set_index("epoch")[["loss", "val_loss"]])

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=-1).numpy()
        embeddings = model.conv2(data.x, data.edge_index).numpy()

    components = PCA(n_components=3).fit_transform(embeddings)
    plot_df = pd.DataFrame(
        {
            "x": components[:, 0],
            "y": components[:, 1],
            "z": components[:, 2],
            "prediction": np.where(predictions == 1, "Illicit", "Licit"),
            "class": np.where(y.numpy() == 1, "Illicit", np.where(y.numpy() == 0, "Licit", "Unknown")),
            "time_step": timestamps.numpy(),
        }
    )
    st.subheader("Interactive 3D transaction visualization")
    fig = px.scatter_3d(
        plot_df,
        x="x",
        y="y",
        z="z",
        color="prediction",
        hover_data=["time_step", "class"],
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Choose a model and click **Train model**.")
