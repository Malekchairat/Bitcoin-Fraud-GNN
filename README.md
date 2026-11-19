# Bitcoin Fraud Detection GNN

Projet de détection de transactions Bitcoin illicites sur le jeu de données
Elliptic. Il compare un MLP, un GraphSAGE et un GAT avec une validation
temporelle qui évite les fuites de données.

## Résultat principal

Une évaluation normale sur **5 seeds** donne les moyennes suivantes :

| Modèle | F1 illicite | Recall illicite | PR-AUC |
|---|---:|---:|---:|
| **GraphSAGE** | **0.5454** | 0.5633 | **0.4967** |
| MLP | 0.4874 | 0.5215 | 0.3105 |
| GAT | 0.4754 | 0.4988 | 0.4133 |

GraphSAGE est actuellement le meilleur compromis global.

## Contenu du projet

```text
Bitcoin-Fraud-GNN/
├── Bitcoin_Fraud_Detection_GNN.ipynb  # Analyse, entraînement et évaluation
├── train_models.py                    # Entraînement MLP, GraphSAGE et GAT
├── export_react_results.py            # Export des résultats vers React
├── run_normal_and_web.ps1             # Lancement rapide du site
├── web/                               # Interface React moderne
├── data/raw/                          # Fichiers Elliptic originaux
├── data/processed/                    # Données légères pour les visualisations
├── outputs/tables/                    # Résultats CSV
├── outputs/models/                    # Poids des modèles entraînés
└── tests/                             # Tests du dashboard
```

Les fichiers complets du dossier `data/raw/` ne sont pas nécessaires pour
afficher le site React. Ils ne sont pas envoyés sur GitHub.

## Démarrage rapide

Depuis PowerShell :

```powershell
cd C:\Users\msi\Desktop\Bitcoin-Fraud-GNN
.\run_normal_and_web.ps1
```

Cette commande :

1. utilise les résultats normaux déjà enregistrés ;
2. exporte les métriques et les prédictions vers React ;
3. démarre l’interface sur `http://localhost:5173`.

Elle ne relance pas les entraînements.

Pour démarrer uniquement React :

```powershell
cd C:\Users\msi\Desktop\Bitcoin-Fraud-GNN\web
npm.cmd install
npm.cmd run dev
```

## Entraîner les modèles

Le mode FAST utilise 1 seed, 30 epochs maximum et une patience de 5.

Le mode normal utilise 5 seeds, 200 epochs maximum et une patience de 30 :

```powershell
cd C:\Users\msi\Desktop\Bitcoin-Fraud-GNN
.\.venv\Scripts\python.exe train_models.py --normal
```

Après l’entraînement, exporter les nouveaux résultats :

```powershell
.\.venv\Scripts\python.exe export_react_results.py
```

Relancer tout avec entraînement normal :

```powershell
.\run_normal_and_web.ps1 -TrainNormal
```

## Exécuter le notebook

Le notebook doit être ouvert depuis le dossier du projet :

```powershell
cd C:\Users\msi\Desktop\Bitcoin-Fraud-GNN
.\.venv\Scripts\Activate.ps1
jupyter lab Bitcoin_Fraud_Detection_GNN.ipynb
```

Dans JupyterLab :

1. vérifier que `FAST_MODE = False` ;
2. choisir **Restart Kernel and Run All Cells** ;
3. ne relancer aucun entraînement supplémentaire si les résultats sont déjà
   enregistrés.

Pour exécuter le notebook avec le script :

```powershell
.\run_normal_and_web.ps1 -ExecuteNotebook
```

## Interface React

L’interface principale se trouve dans `web/`.

Elle contient :

- une vue générale du dataset ;
- la distribution des classes ;
- le découpage temporel ;
- les résultats des trois modèles ;
- un cube réseau interactif ;
- les relations réelles entre transactions ;
- des prédictions MLP, GraphSAGE et GAT sur les transactions inconnues.

### Utiliser le graphe

- faire défiler ou pincer pour zoomer ;
- glisser pour tourner le réseau ;
- cliquer sur un nœud pour voir ses détails ;
- modifier le nombre de nœuds affichés ;
- sélectionner **Original labels** pour laisser les inconnus en gris ;
- sélectionner MLP, GraphSAGE ou GAT pour afficher ses prédictions.

Les prédictions changent uniquement les transactions actuellement marquées
comme inconnues. Les transactions licites ou illicites connues ne sont pas
modifiées.

## Construction du site React

```powershell
cd C:\Users\msi\Desktop\Bitcoin-Fraud-GNN\web
npm.cmd install
npm.cmd run build
npm.cmd run preview
```

## Déploiement sur Vercel

Le dossier public pour Vercel est `web/`.

Dans Vercel :

- Root Directory : `web`
- Framework : Vite
- Build Command : `npm run build`
- Output Directory : `dist`

Les résultats affichés proviennent de fichiers JSON exportés depuis les
résultats réels du notebook. Ils ne sont pas générés aléatoirement.

## Déploiement avec Docker

```powershell
cd C:\Users\msi\Desktop\Bitcoin-Fraud-GNN
docker build -t bitcoin-fraud-dashboard .
docker run --rm -p 8080:80 bitcoin-fraud-dashboard
```

Puis ouvrir `http://localhost:8080`.

## Tests

```powershell
cd C:\Users\msi\Desktop\Bitcoin-Fraud-GNN
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Informations importantes

- `unknown` signifie « étiquette non connue », pas licite.
- Les mesures principales sont F1 illicite, recall, ROC-AUC et PR-AUC.
- Le découpage est temporel : entraînement sur les steps 1–29, validation sur
  les steps 30–34, test sur les steps 35–49.
- Les métriques normal-mode sont calculées sur 5 seeds.
- Le graphe React est un échantillon explicatif de 2 000 transactions ; il ne
  dessine pas les 203 769 transactions simultanément.
