# Wi-Fi CSI HAR — Experimental Codebase

Code Python pour la partie **expérimentation** de l'article :

**Prototype-Consistent and Reliability-Calibrated Domain Generalization for Wi-Fi CSI Human Activity Recognition**

## Ce que contient le projet

- `configs/` : fichiers YAML pour NTU-Fi, Widar 3.0 et un format générique
- `scripts/train.py` : entraînement
- `scripts/eval.py` : évaluation d'un checkpoint
- `scripts/make_splits.py` : génération des splits expérimentaux
- `scripts/inspect_dataset.py` : inspection rapide des métadonnées
- `src/wificsi_exp/` : code source principal
- `tests/` : tests simples

## Contributions implémentées

Le code prend en charge :

1. **ERM** (baseline)
2. **Prototype consistency**
3. **Supervised contrastive learning**
4. **Calibration-aware learning (Brier)**
5. **Selective prediction / rejection** à l'inférence

## Compatibilité

- Compatible **Windows**
- Utilise `pathlib`
- Gère des structures de datasets différentes via des **adapters** et un fichier `metadata.csv`

## Format attendu des données


Chaque dataset doit fournir un `metadata.csv` avec au minimum les colonnes suivantes :

- `sample_path` : chemin vers le fichier `.npy` ou `.npz`
- `label` : nom ou id de l'activité
- `domain` : domaine principal pour le protocole DG

Colonnes optionnelles :

- `subject`
- `environment`
- `orientation`
- `split`

Le tenseur CSI attendu par défaut est de forme :

- `(C, S, T)`

avec par exemple :
- `C = 3`
- `S = 114`
- `T = 500`

### Si les structures diffèrent selon les datasets
Le projet utilise des **adapters** :
- `GenericMetadataDataset`
- `NTUFiDataset`
- `Widar3Dataset`

Si ton dataset a une structure spéciale, adapte surtout :

- `src/wificsi_exp/data/adapters.py`

## Installation

### 1. Créer l'environnement

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Sous CMD :

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
```

## Commandes utiles

### Inspecter un dataset

```powershell
python scripts\inspect_dataset.py --config configs\ntu_fi.yaml
```

### Générer des splits

```powershell
python scripts\make_splits.py --config configs\widar3.yaml --protocol leave-one-domain-out
```

### Entraîner

```powershell
python scripts\train.py --config configs\widar3.yaml --run_name proto_dg_exp
```

### Évaluer

```powershell
python scripts\eval.py --config configs\widar3.yaml --checkpoint runs\proto_dg_exp\best.pt
```

## Baselines supportées

Dans le fichier de config :

- `erm`
- `proto`
- `supcon`
- `brier`
- `proto_supcon_brier`

La selective prediction est contrôlée séparément à l'évaluation.

## Métriques produites

- Accuracy
- Macro-F1
- ECE
- Brier Score
- Coverage
- Selective Accuracy
- Risk-Coverage table

## Organisation conseillée des expériences

1. NTU-Fi en in-domain
2. Widar 3.0 en leave-one-domain-out
3. Ablations :
   - ERM
   - ERM + Proto
   - ERM + SupCon
   - ERM + Brier
   - Full model
4. Évaluation selective prediction

## Remarques importantes

- Les seuils de rejet `tau_conf` et `tau_dist` doivent être choisis sur validation uniquement.
- Le domaine de test ne doit jamais servir à régler les hyperparamètres.
- Il est recommandé de faire plusieurs seeds.

## Structure du projet

```text
wifi_csi_article_experimentation/
├── configs/
├── scripts/
├── src/wificsi_exp/
│   ├── data/
│   ├── engine/
│   ├── losses/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── tests/
└── README.md
```
