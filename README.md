# Bitcoin-Fraud-GNN
# Bitcoin Fraud Detection with Graph Neural Networks

This project demonstrates fraud detection on the Elliptic Bitcoin transaction
dataset using GraphSAGE and Graph Attention Networks (GAT). It includes temporal
train/validation/test splits, standard evaluation metrics, and an interactive
3D Streamlit visualization.

## Setup

```bash
pip install -r requirements.txt
```

Download the three Elliptic CSV files and place them in `data/`:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

## Run the dashboard

```bash
streamlit run app.py
```

Select a model and click **Train model**. The dashboard reports accuracy, F1,
ROC-AUC, average precision, and displays transactions in 3D using PCA on model
embeddings.

## Train from the command line

```bash
python train.py data --model GraphSAGE --epochs 100 --output graphsage.pt
python train.py data --model GAT --epochs 100 --output gat.pt
```