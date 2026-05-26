# AutoML Studio

Application web pour l’**apprentissage automatique** : import CSV, préparation des données, entraînement multi-modèles, suivi **MLflow**, historique des expériences et **rapport d’analyse** pédagogique (4 questions, graphiques matplotlib).

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)

---

## Démarrage rapide

```powershell
# 1. Backend
cd automl-backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python run_dev.py

# 2. Frontend (autre terminal)
cd automl-frontend
npm install
npm start

# 3. MLflow UI (optionnel, 3e terminal — même URI que le backend)
cd automl-backend
.\venv\Scripts\activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

| Service | URL |
|---------|-----|
| Interface | http://localhost:3000 (ou **3001** si 3000 occupé) |
| API + Swagger | http://localhost:8000 · http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

**Connexion par défaut** (créée au 1er démarrage si aucun utilisateur) : `admin@automl.local` / `admin123`

> Utilisez **`python run_dev.py`** et non `uvicorn --reload` seul : sinon chaque écriture MLflow dans `mlruns/` redémarre le serveur pendant l’entraînement.

---

## Table des matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Lancement](#lancement)
- [Authentification JWT](#authentification-jwt)
- [Parcours utilisateur](#parcours-utilisateur)
- [Rapport d’analyse](#rapport-danalyse-4-questions)
- [Historique et statistiques](#historique-et-statistiques)
- [Structure du projet](#structure-du-projet)
- [API REST](#api-rest)
- [Algorithmes](#algorithmes)
- [Variables d’environnement](#variables-denvironnement)
- [Dépannage](#dépannage)
- [Publication Git](#publication-git)
- [Licence](#licence)

---

## Aperçu

**AutoML Studio** enchaîne :

1. **Upload** d’un CSV (classification ou régression)
2. **Nettoyage** (doublons, NA, outliers, normalisation)
3. **Configuration** (cible, features, algorithmes, split, expérience MLflow)
4. **Entraînement** multi-modèles avec métriques et tracking
5. **Comparaison** des résultats + export CSV
6. **Rapport** visuel pour le meilleur modèle (4 questions pédagogiques)

Stack : **React 19** · **FastAPI** · **scikit-learn** · **MLflow 2.13** · **JWT (bcrypt)**.

---

## Fonctionnalités

### Interface React

| Étape | Description |
|--------|-------------|
| **Login** | Inscription / connexion JWT, mode dev sans auth possible |
| **Upload** | CSV, aperçu, stats colonnes, types détectés |
| **Nettoyage** | Doublons, imputation, outliers (IQR / Z-score), scaling, suppression colonnes |
| **Configuration** | Cible, tâche, features, `test_size`, algorithmes, nom d’expérience |
| **Résultats** | Meilleur modèle, métriques, comparaison, export |
| **Rapport** | Modal 4 questions (importance, stabilité, erreurs, biais/variance) |
| **Panneaux** | MLflow intégré, **historique** des runs + onglet **statistiques** |

### Back-end FastAPI

- Datasets persistés en `automl-backend/datasets/` (survit au redémarrage)
- **MLflow** : paramètres, métriques, artefacts modèle (`artifact_path=model_{algo}`)
- Réparation auto d’une `mlflow.db` corrompue au démarrage (`bootstrap_mlflow_client`)
- Sous-échantillonnage SVM/SVR au-delà de 5 000 lignes (`MAX_ROWS_FOR_SVM`)
- Maintenance : purge runs / reset store MLflow
- **CORS** : ports 3000/3001 + regex localhost pour le dev

### Sécurité

- JWT HS256, mots de passe **bcrypt**
- Routes ML protégées ; `/health`, `/auth/*` publics
- Compte admin auto si `data/users.json` vide
- `AUTH_DISABLED=true` pour développement sans login

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Navigateur — localhost:3000 ou :3001                       │
│  React : Login → Upload → Preprocess → Config → Dashboard   │
└────────────────────────────┬────────────────────────────────┘
                             │ REST + Bearer JWT
┌────────────────────────────▼────────────────────────────────┐
│  FastAPI — localhost:8000                                   │
│  main.py · auth.py · report_* · mlflow_utils.py · run_dev.py│
└────────────────────────────┬────────────────────────────────┘
                             │ MLflow Tracking API
┌────────────────────────────▼────────────────────────────────┐
│  MLflow UI — localhost:5000                                 │
│  sqlite:///mlflow.db (ou mlflow_fresh.db) · mlruns/         │
└─────────────────────────────────────────────────────────────┘
```

---

## Prérequis

| Outil | Version | Vérification |
|--------|---------|--------------|
| **Python** | **3.11.x** recommandé | `python --version` |
| **Node.js** | 18 LTS ou 20+ | `node --version` |
| **npm** | 9+ | `npm --version` |

> Éviter Python 3.12+ si des incompatibilités apparaissent avec numpy / scikit-learn / MLflow.

---

## Installation

### 1. Cloner

```bash
git clone <URL_DU_REPO>
cd automl-studio-complet
```

### 2. Configuration (optionnel)

```powershell
copy .env.example automl-backend\.env
# Éditer JWT_SECRET, INIT_ADMIN_*, MLFLOW_*, etc.
```

### 3. Back-end

```powershell
cd automl-backend
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Dépendances clés** : `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `mlflow==2.13.0`, `matplotlib`, `setuptools<81` (pour `pkg_resources`), `python-jose`, `passlib`, `bcrypt`.

### 4. Front-end

```powershell
cd ..\automl-frontend
npm install
```

Variable optionnelle : `REACT_APP_API_URL=http://localhost:8000` (fichier `.env` dans `automl-frontend/`).

---

## Lancement

### Terminal 1 — API (recommandé)

```powershell
cd automl-backend
.\venv\Scripts\activate
python run_dev.py
```

`run_dev.py` lance uvicorn avec `--reload` en **excluant** `mlruns/`, `datasets/`, `*.db` du file-watcher.

### Terminal 2 — MLflow UI

Utiliser la **même URI** que celle loguée au démarrage du backend (souvent `sqlite:///mlflow.db`, parfois `sqlite:///mlflow_fresh.db` si l’ancienne base est verrouillée) :

```powershell
cd automl-backend
.\venv\Scripts\activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

### Terminal 3 — React

```powershell
cd automl-frontend
npm start
```

Si le port 3000 est pris, React propose **3001** — le CORS backend l’accepte par défaut.

---

## Authentification JWT

### Compte par défaut

| Champ | Valeur |
|--------|--------|
| Email | `admin@automl.local` (ou `INIT_ADMIN_EMAIL`) |
| Mot de passe | `admin123` (ou `INIT_ADMIN_PASSWORD`) |

Créé automatiquement si `automl-backend/data/users.json` n’existe pas.

### Endpoints publics

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/auth/config` | Auth activée ou non |
| `POST` | `/auth/register` | Inscription → JWT |
| `POST` | `/auth/login` | Connexion → JWT |

### Routes protégées

Header requis :

```http
Authorization: Bearer <access_token>
```

Exemples : `/upload`, `/preprocess`, `/train`, `/report`, `/experiments`, `/stats`.

### Exemple curl

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"admin@automl.local\",\"password\":\"admin123\"}"
```

### Fichiers auth

| Fichier | Rôle |
|---------|------|
| `automl-backend/auth.py` | JWT, utilisateurs JSON |
| `automl-backend/data/users.json` | Comptes (gitignore) |
| `automl-frontend/src/authStorage.js` | Token localStorage |
| `automl-frontend/src/components/LoginPage.jsx` | UI connexion |

---

## Parcours utilisateur

1. **Connexion** (ou `AUTH_DISABLED=true`)
2. **Upload** — CSV, aperçu
3. **Nettoyage** — options puis validation
4. **Configuration** — cible, tâche, features, algorithmes, expérience MLflow
5. **Résultats** — entraînement automatique au chargement du dashboard
6. **Rapport** — bouton sur le meilleur modèle (timeout front ~120 s)
7. **Historique / MLflow** — panneaux depuis l’en-tête

### Choisir la cible

| Profil | Tâche |
|--------|--------|
| Peu de classes / booléen / catégoriel | Classification |
| Numérique continue | Régression |
| Identifiant unique par ligne | **Ne pas** utiliser comme cible |

---

## Rapport d’analyse (4 questions)

Généré pour le **meilleur modèle** (`best_algo_id` après `/train`).

| # | Thème | Contenu |
|---|--------|---------|
| Q1 | Importance des features | Barres + tableau + interprétation |
| Q2 | Stabilité | Variation selon `random_state` |
| Q3 | Erreurs | Matrice de confusion / scatter, exemples |
| Q4 | Biais / variance | Courbes train vs test, tableau overfitting |

Export Markdown depuis le modal.  
Backend : `report_generator.py`, `report_visuals.py` — Front : `ReportPanel.jsx`, `reportPayload.js`.

---

## Historique et statistiques

Le panneau **Historique** (`Historypanel.jsx`) affiche les runs MLflow de l’expérience sélectionnée :

- **Runs** : algorithme (icône + nom), tâche, cible, métriques, filtre / tri / export CSV
- **Models** : registry MLflow (si activé)
- **Statistiques** : runs par algorithme, classification vs régression, totaux globaux (`GET /stats`)

L’API renvoie pour chaque run :

- `algo` — id court (`rf`, `svm`, `dt`…)
- `algo_name` — libellé affiché (`Random Forest`, etc.)

Libellés partagés : `automl-frontend/src/algoMeta.js`.

---

## Structure du projet

```
automl-studio-complet/
├── README.md
├── .gitignore
├── .env.example
│
├── automl-backend/
│   ├── main.py                 # API FastAPI
│   ├── auth.py                 # JWT
│   ├── run_dev.py              # Lancement dev (reload safe)
│   ├── mlflow_utils.py         # Bootstrap DB, cleanup, reset
│   ├── report_generator.py
│   ├── report_visuals.py
│   ├── requirements.txt
│   ├── datasets/               # CSV persistés (gitignore)
│   ├── data/users.json         # Comptes (gitignore)
│   ├── mlflow.db               # Généré (gitignore)
│   └── mlruns/                 # Artefacts MLflow (gitignore)
│
└── automl-frontend/
    ├── package.json
    └── src/
        ├── App.jsx
        ├── api.js
        ├── authStorage.js
        ├── algoMeta.js
        ├── reportPayload.js
        └── components/
            ├── LoginPage.jsx
            ├── UploadStep.jsx
            ├── PreprocessStep.jsx
            ├── ConfigStep.jsx
            ├── Dashboard.jsx
            ├── ReportPanel.jsx
            ├── MLflowPanel.jsx
            └── Historypanel.jsx
```

---

## API REST

| Méthode | Endpoint | Auth | Description |
|---------|----------|------|-------------|
| `GET` | `/` | — | Infos API |
| `GET` | `/health` | — | Santé + espace disque |
| `POST` | `/auth/register` | — | Inscription |
| `POST` | `/auth/login` | — | Connexion |
| `GET` | `/auth/me` | JWT | Profil |
| `GET` | `/auth/config` | — | Config auth |
| `POST` | `/upload` | JWT | Upload CSV |
| `POST` | `/preprocess` | JWT | Nettoyage |
| `POST` | `/train` | JWT | Entraînement multi-algos |
| `POST` | `/report` | JWT | Rapport d’analyse |
| `POST` | `/predict` | JWT | Prédiction via `run_id` |
| `GET` | `/experiments` | JWT | Liste expériences |
| `GET` | `/experiments/{name}/runs` | JWT | Runs formatés (`algo`, `algo_name`) |
| `GET` | `/models` | JWT | Model Registry |
| `GET` | `/stats` | JWT | Totaux, runs par algo, best accuracy/R² |
| `POST` | `/maintenance/cleanup` | JWT | Purge anciens runs |
| `POST` | `/maintenance/reset` | JWT | Reset MLflow (destructif) |

Documentation interactive : **http://localhost:8000/docs**

### Exemple `POST /train`

```json
{
  "dataset_id": "ds_1778494906",
  "task_type": "classification",
  "target": "city_tier",
  "features": ["age", "order_value"],
  "algorithms": ["rf", "svm", "dt"],
  "test_size": 0.2,
  "experiment_name": "AutoML_Studio"
}
```

---

## Algorithmes

### Classification

| ID | Nom |
|----|-----|
| `rf` | Random Forest |
| `svm` | SVM |
| `lr` | Logistic Regression |
| `knn` | KNN |
| `dt` | Decision Tree |
| `nb` | Naive Bayes |

### Régression

| ID | Nom |
|----|-----|
| `rf` | Random Forest |
| `lr` | Linear Regression |
| `ridge` | Ridge |
| `lasso` | Lasso |
| `svr` | SVR |
| `dt` | Decision Tree |

### Métriques

- **Classification** : accuracy, F1, precision, recall, AUC (binaire), CV, matrice de confusion, temps
- **Régression** : R², RMSE, MAE, CV R², temps

---

## Variables d’environnement

Fichier modèle : **`.env.example`** (copier vers `automl-backend/.env`).

| Variable | Défaut | Description |
|----------|--------|-------------|
| `JWT_SECRET` | *(à changer)* | Clé signature JWT |
| `JWT_EXPIRE_MINUTES` | `1440` | Durée token (min) |
| `INIT_ADMIN_EMAIL` | `admin@automl.local` | Admin initial |
| `INIT_ADMIN_PASSWORD` | `admin123` | Mot de passe admin |
| `AUTH_DISABLED` | `false` | Désactiver JWT (dev) |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | URI tracking |
| `MLFLOW_MAX_RUNS` | `20` | Runs max / expérience |
| `MLFLOW_MAX_MODEL_VERSIONS` | `3` | Versions registry |
| `MLFLOW_REGISTER_MODELS` | `false` | Enregistrer dans le registry |
| `MAX_ROWS_FOR_SVM` | `5000` | Limite lignes SVM/SVR |
| `CORS_ORIGINS` | *(liste par défaut)* | Origines explicites |
| `CORS_ALLOW_ORIGIN_REGEX` | `localhost:\d+` | Regex dev |
| `REACT_APP_API_URL` | `http://localhost:8000` | URL API (frontend) |

---

## Dépannage

### API injoignable

```powershell
cd automl-backend
.\venv\Scripts\activate
python run_dev.py
```

Vérifier : http://localhost:8000/health

### `Failed to fetch` / CORS (login ou train)

- Redémarrer le backend après mise à jour.
- Front sur **3001** : accepté par défaut.
- Pendant **train** : utiliser **`python run_dev.py`**, pas `uvicorn --reload` seul.
- Gros CSV : entraînement long (timeout front train : 15 min).

### `log_model() got an unexpected keyword argument 'name'`

MLflow 2.13 utilise **`artifact_path`**, pas `name` — corrigé dans `main.py` ; redémarrer le backend.

### `No module named 'pkg_resources'`

```powershell
.\venv\Scripts\pip.exe install "setuptools>=65.5.0,<81"
pip install -r requirements.txt
```

### MLflow : `Can't locate revision` / DB corrompue

1. Arrêter backend + `mlflow ui`.
2. Supprimer `mlflow.db`, `mlflow.db-wal`, `mlflow.db-shm`, optionnellement `mlruns/`.
3. Relancer `python run_dev.py`.

Ou `POST /maintenance/reset` (JWT requis). Si bascule vers `mlflow_fresh.db`, aligner `mlflow ui --backend-store-uri sqlite:///mlflow_fresh.db`.

### Historique : « unknown » / statistiques vides

- Vérifier que les runs ont `params.algorithm` (ré-entraîner après mise à jour).
- Choisir la bonne expérience (onglet `AutoML_Studio`).
- Même URI MLflow entre backend et UI.

### Connexion `422` / email invalide

Email `admin@automl.local` accepté (validation assouplie). Vérifier API + `REACT_APP_API_URL`.

### `401 Unauthorized`

Reconnecter ; token expiré. Ou `AUTH_DISABLED=true` en dev.

### `Dataset non trouvé`

Refaire upload → preprocess → train, ou vérifier `datasets/{dataset_id}.csv`.

### Disque plein

`POST /maintenance/cleanup` ou supprimer `mlruns/` manuellement.

---

## Publication Git

### À ne **pas** committer

- `automl-backend/venv/`, `__pycache__/`
- `automl-backend/mlruns/`, `mlflow.db`, `mlflow_fresh.db`
- `automl-backend/datasets/*.csv`
- `automl-backend/data/users.json`
- `automl-frontend/node_modules/`, `build/`
- `.env`, secrets, données personnelles

### Avant le push

1. `JWT_SECRET` fort en production (pas la valeur exemple).
2. Changer `INIT_ADMIN_PASSWORD` après le 1er login.
3. Choisir une **licence** (section ci-dessous).
4. Remplacer `<URL_DU_REPO>` dans ce README par l’URL Git réelle.

### Suggestion de message de commit initial

```text
feat: AutoML Studio — React + FastAPI + MLflow + rapport JWT
```

---

## Contribution

1. Fork du dépôt  
2. Branche : `git checkout -b feature/ma-fonctionnalite`  
3. Commits clairs  
4. Pull request vers `main`

---

## Licence

Projet académique / personnel — **à préciser** avant publication (ex. MIT, Apache-2.0).

---

**AutoML Studio** — React · FastAPI · scikit-learn · MLflow · JWT · Rapport d’analyse intégré
