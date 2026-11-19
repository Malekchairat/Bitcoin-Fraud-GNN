# ₿ Détection de fraude Bitcoin avec les Graph Neural Networks

**Explorer les liens entre transactions Bitcoin pour mieux détecter les activités illicites grâce au Machine Learning et aux réseaux de neurones sur graphes.**

Ce projet s'appuie sur le jeu de données **Elliptic Bitcoin Dataset** pour étudier la détection de transactions suspectes à travers différentes approches d'apprentissage automatique.

Il compare trois modèles — **MLP, GraphSAGE et GAT** — et propose un tableau de bord interactif pour explorer les transactions, leurs connexions et les prédictions des modèles.

## ✨ Fonctionnalités

- **Exploration des données :** analyse des transactions, des classes et de la structure du réseau.
- **Machine Learning :** entraînement et comparaison de trois architectures.
- **Validation temporelle :** respect de l'ordre chronologique des transactions pour éviter les fuites de données.
- **Visualisation interactive :** exploration du réseau de transactions et de leurs relations.
- **Analyse des prédictions :** comparaison des classifications produites par les modèles.

## 🛠️ Technologies utilisées

`Python` · `PyTorch` · `PyTorch Geometric` · `Pandas` · `Scikit-learn` · `React` · `Vite` · `JavaScript`

## 📁 Structure du projet

```text
Bitcoin-Fraud-GNN/
│
├── Bitcoin_Fraud_Detection_GNN.ipynb
├── train_models.py
├── export_react_results.py
├── run_normal_and_web.ps1
│
├── web/                  # Interface React
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── models/
│   └── tables/
│
└── tests/
```

## 🚀 Installation et démarrage

**1. Cloner le projet**

```bash
git clone https://github.com/Malekchairat/Bitcoin-Fraud-GNN.git
cd Bitcoin-Fraud-GNN
```

**2. Installer les dépendances**

Activer l'environnement Python et installer les dépendances :

```bash
pip install -r requirements.txt
```

Installer celles de l'interface React :

```bash
cd web
npm install
```

**3. Lancer le tableau de bord**

Depuis le dossier principal du projet :

```powershell
.\run_normal_and_web.ps1
```

Pour lancer uniquement l'interface React :

```bash
cd web
npm run dev
```

Ouvrir ensuite l'adresse locale affichée dans le terminal.

## 🧠 Entraînement des modèles

Lancer l'entraînement complet :

```bash
python train_models.py --normal
```

Exporter les résultats pour le tableau de bord :

```bash
python export_react_results.py
```

## 📊 Tableau de bord interactif

L'interface permet de :

- Explorer les données et la répartition des transactions.
- Visualiser les connexions entre les transactions Bitcoin.
- Parcourir le réseau à travers un graphe interactif.
- Consulter les détails de chaque nœud.
- Comparer les prédictions de MLP, GraphSAGE et GAT.

## 📌 À noter

- Une transaction inconnue ne signifie pas nécessairement qu'elle est licite.
- Le jeu de données suit une organisation chronologique.
- Le graphe affiché dans l'interface est un échantillon du réseau complet.
- Les données brutes ne sont pas incluses dans le dépôt GitHub.

---

*Un projet à la croisée du Machine Learning, des Graph Neural Networks et de la détection de fraude financière.*
