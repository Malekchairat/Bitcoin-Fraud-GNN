# Bitcoin Fraud Detection GNN

This project studies illicit Bitcoin transactions in the Elliptic transaction
graph. The notebook builds the graph and compares graph neural network models
with a non-graph baseline using a leakage-safe temporal split.

## View the results

Install the pinned dependencies, then start the dashboard from the project
root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard opens in a browser and shows:

- transaction, edge, feature, and class totals;
- the notebook's chronological train/validation/test split;
- transaction volume and illicit-rate charts over time;
- a transaction ID lookup;
- a labelled graph view of a transaction and its direct neighbors;
- a rotatable, zoomable 3D WebGL graph overview with performance controls;
- saved model comparison tables from `outputs/tables`, when available.

The app does not retrain models automatically. Run the notebook first if you
want model metrics to appear in the final section.

## Execution modes

The notebook now sets `FAST_MODE = False`, so it is aligned with normal mode:

- Normal: 200 epochs maximum, patience 30, 5 seeds
- FAST: 30 epochs maximum, patience 5, 1 seed
- The external `--balanced` option: 60 epochs maximum, patience 10, 3 seeds

The five-seed normal run is the most robust current result.

## Polished public interface

A React/Vite interface is available in [web/](C:/Users/msi/Desktop/Bitcoin-Fraud-GNN/web).
It is not a duplicate Streamlit prototype: it presents the research story,
dataset, graph, and normal-mode model comparison with a cleaner navigation and
responsive layout.

For local development:

```powershell
cd web
npm.cmd install
npm.cmd run dev
```

For a production build:

```powershell
cd web
npm.cmd run build
npm.cmd run preview
```

The React interface is ready for Vercel. In the Vercel project settings, set:

- Root Directory: `web`
- Framework Preset: Vite
- Build Command: `npm run build`
- Output Directory: `dist`

## Deploy on Streamlit Community Cloud

This app is ready for Streamlit Community Cloud:

1. Push this project to a public or private GitHub repository.
2. Open Streamlit Community Cloud and create a new app.
3. Select the repository and set the main file to `app.py`.
4. Set Python 3.13 if the platform asks for a version.
5. Add no secrets: this dashboard reads local processed CSV files.

The repository should include `app/`, `data/processed/`, and `outputs/tables/`.
The raw 690 MB feature file is not required for the public dashboard and is
excluded through `.gitignore`.

### Deploy with Docker

Docker is also supported:

```powershell
docker build -t bitcoin-fraud-dashboard .
docker run --rm -p 8501:8501 bitcoin-fraud-dashboard
```

Vercel is not the simplest platform for this application because Streamlit is
a stateful Python server. Streamlit Community Cloud or Render are easier
choices. Docker deployment works on Azure Container Apps or another container
platform later.

## Train the models

The modelling phase is also available as a reproducible script. It trains the
MLP baseline, GraphSAGE, and GAT using the same temporal split as the notebook:

```powershell
.\.venv\Scripts\python.exe train_models.py
```

The default FAST run uses one seed, up to 30 epochs, and early stopping. It
writes test metrics to `outputs/tables/model_results.csv`, training traces to
`outputs/tables/training_history.csv`, threshold searches to
`outputs/tables/threshold_search.csv`, and model weights to `outputs/models`.

For the notebook-normal experiment with five seeds, up to 200 epochs, and
patience 30:

```powershell
.\.venv\Scripts\python.exe train_models.py --normal
```

For a lower-CPU multi-seed experiment with three seeds, up to 60 epochs, and
patience 10:

```powershell
.\.venv\Scripts\python.exe train_models.py --balanced
```