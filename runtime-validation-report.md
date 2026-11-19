# Runtime Validation Report

**Target**: Bitcoin-Fraud-GNN

## Summary

| Check | Status | Details |
|---|---|---|
| Python syntax | PASS | `app.py`, `train_models.py`, and dashboard tests compile |
| Dependency check | PASS | `pip check`: no broken requirements |
| Streamlit AppTest | PASS | 3 dashboard tests passed |
| Normal training | PASS | 5 seeds, 200 epochs maximum, patience 30 |
| Docker | UNAVAILABLE | Docker executable found, but the daemon is not running |
| Node.js | AVAILABLE | Node.js 22.17.1 |

## Normal model results

The final table contains 15 rows: 5 seeds for MLP, GraphSAGE, and GAT.

| Model | Mean F1 | F1 Std | Mean PR-AUC | PR-AUC Std |
|---|---:|---:|---:|---:|
| GraphSAGE | 0.5454 | 0.0387 | 0.4967 | 0.0353 |
| MLP | 0.4874 | 0.0198 | 0.3105 | 0.0159 |
| GAT | 0.4754 | 0.0268 | 0.4133 | 0.0790 |

GraphSAGE is the best overall normal-mode model by mean illicit F1 and PR-AUC.

## Dashboard validation

Verified flows:

- dashboard loads without exceptions;
- transaction lookup accepts transaction `230425980`;
- graph controls are present;
- processed deployment data contains 203,769 transactions and 234,355 edges.

## Deployment verdict

The easiest public deployment is **Streamlit Community Cloud**. Vercel is not
the preferred host for this stateful Python application. Docker deployment is
also prepared, but local Docker validation was unavailable because the daemon
is stopped.
