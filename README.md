# ⚗️ AutoML Studio

> **Application professionnelle de Machine Learning** avec upload de dataset, entraînement automatique, comparaison de modèles et tracking MLflow.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2?logo=mlflow)

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Lancement](#lancement)
- [Fonctionnalités](#fonctionnalités)
- [Structure du projet](#structure-du-projet)
- [API Endpoints](#api-endpoints)
- [Algorithmes disponibles](#algorithmes-disponibles)
- [Utilisation](#utilisation)
- [Dépannage](#dépannage)

---

## 🎯 Aperçu

AutoML Studio est une application web complète qui permet à n'importe quel utilisateur de :

1. **Uploader** n'importe quel dataset CSV
2. **Configurer** le type de tâche (Classification ou Régression)
3. **Entraîner** plusieurs algorithmes ML en parallèle
4. **Comparer** les résultats avec des graphiques interactifs
5. **Tracker** toutes les expériences avec MLflow

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    UTILISATEUR                       │
│                  localhost:3000                      │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP REST
┌──────────────────────▼──────────────────────────────┐
│               REACT FRONT-END                        │
│  UploadStep → ConfigStep → Dashboard → MLflowPanel  │
└──────────────────────┬──────────────────────────────┘
                       │ fetch() API calls
┌──────────────────────▼──────────────────────────────┐
│              FASTAPI BACK-END                        │
│              localhost:8000                          │
│   /upload  /train  /experiments  /models  /stats    │
└──────────────────────┬──────────────────────────────┘
                       │ Python SDK
┌──────────────────────▼──────────────────────────────┐
│                  MLFLOW                              │
│              localhost:5000                          │
│     SQLite DB │ Experiments │ Runs │ Model Registry │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ Prérequis

| Outil | Version minimale | Vérification |
|-------|-----------------|--------------|
| Python | 3.11 | `py --list` |
| Node.js | 16+ | `node --version` |
| npm | 8+ | `npm --version` |

> ⚠️ **Important** : Utiliser **Python 3.11** uniquement. Python 3.14 est incompatible avec certaines dépendances ML.

---

## 🚀 Installation

### 1. Cloner / Extraire le projet

```
automl-studio-complet/
├── automl-backend/
└── ml-generic/
```

### 2. Installer le Back-End Python

```powershell
# Aller dans le dossier backend
cd automl-backend

# Créer un environnement virtuel Python 3.11
py -3.11 -m venv venv

# Activer l'environnement
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Mettre à jour pip
python.exe -m pip install --upgrade pip setuptools wheel

# Installer les dépendances
pip install fastapi uvicorn pydantic pandas numpy scikit-learn mlflow joblib python-multipart
```

### 3. Installer le Front-End React

```powershell
# Aller dans le dossier racine
cd ..

# Créer le projet React
npx create-react-app automl-frontend

# Copier les fichiers sources
xcopy /E /Y "ml-generic\src\*" "automl-frontend\src\"

# Supprimer les fichiers par défaut de React
cd automl-frontend\src
del App.js App.test.js logo.svg reportWebVitals.js setupTests.js

# Corriger index.js (voir section Utilisation)
```

---

## ▶️ Lancement

Ouvrir **3 terminaux** dans VS Code :

### Terminal 1 — FastAPI (Back-End)

```powershell
cd automl-backend
venv\Scripts\activate
uvicorn main:app --reload
```

✅ Résultat attendu :
```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Started reloader process
```

### Terminal 2 — MLflow UI

```powershell
cd automl-backend
venv\Scripts\activate
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

✅ Résultat attendu :
```
[INFO] Starting gunicorn
[INFO] Listening at: http://127.0.0.1:5000
```

### Terminal 3 — React (Front-End)

```powershell
cd automl-frontend
npm start
```

✅ Résultat attendu : ouverture automatique de `http://localhost:3000`

---

## 🌐 URLs des services

| Service | URL | Description |
|---------|-----|-------------|
| ⚛️ Application | http://localhost:3000 | Interface utilisateur React |
| ⚡ API FastAPI | http://localhost:8000 | Back-End REST |
| 📖 API Docs | http://localhost:8000/docs | Documentation Swagger auto |
| 🔬 MLflow UI | http://localhost:5000 | Tableau de bord MLflow |

---

## ✨ Fonctionnalités

### 📁 Upload de Dataset
- Drag & drop ou sélection de fichier CSV
- Compatible avec **n'importe quel dataset** CSV
- Encodages supportés : UTF-8, Latin-1, CP1252, ISO-8859-1
- Aperçu des 10 premières lignes
- Statistiques automatiques (lignes, colonnes, valeurs nulles)
- **Auto-détection** du type de chaque colonne

### ⚙️ Configuration Intelligente
- Sélection de la **variable cible** avec analyse automatique
- **Auto-détection** Classification vs Régression
- Affichage du nombre de valeurs uniques par colonne
- Sélection multi-features avec filtre numérique/catégoriel
- Choix du split Train/Test (10% à 40%)
- Nom de l'expérience MLflow personnalisable

### 🤖 Entraînement ML
- **6 algorithmes de Classification** : Random Forest, SVM, Logistic Regression, KNN, Decision Tree, Naive Bayes
- **6 algorithmes de Régression** : Random Forest, Linear Regression, Ridge, Lasso, SVR, Decision Tree
- Encodage automatique des variables catégorielles
- Gestion des valeurs manquantes
- Normalisation StandardScaler
- Cross-validation 5-fold

### 📊 Résultats & Visualisations
- **Métriques Classification** : Accuracy, F1-Score, Precision, Recall, AUC-ROC, CV Score
- **Métriques Régression** : R², RMSE, MAE, Train RMSE, CV R²
- Graphique de comparaison des modèles
- Analyse Overfitting (Train vs Test)
- Feature Importance (pour Random Forest et Decision Tree)
- Matrice de Confusion interactive
- Tableau comparatif exportable en CSV

### 🔬 MLflow Integration
- Tracking automatique de **tous les runs**
- Logging des paramètres, métriques et artefacts
- **Model Registry** avec versioning
- Panel MLflow intégré dans l'interface React
- Rollback vers versions précédentes
- Lien direct vers MLflow UI

---

## 📁 Structure du projet

```
automl-studio-complet/
│
├── automl-backend/                 ← Back-End Python
│   ├── main.py                     ← API FastAPI + logique ML
│   ├── requirements.txt            ← Dépendances Python
│   ├── README.md                   ← Ce fichier
│   ├── mlflow.db                   ← Base de données MLflow (auto-créée)
│   └── venv/                       ← Environnement virtuel (à créer)
│
└── ml-generic/                     ← Sources Front-End React
    └── src/
        ├── App.jsx                 ← Composant principal + navigation
        ├── App.css                 ← Styles globaux (thème vert médical)
        ├── api.js                  ← Couche d'appels API vers FastAPI
        └── components/
            ├── UploadStep.jsx      ← Étape 1 : Upload CSV
            ├── ConfigStep.jsx      ← Étape 2 : Configuration ML
            ├── Dashboard.jsx       ← Étape 3 : Résultats & graphiques
            └── MLflowPanel.jsx     ← Panel latéral MLflow
```

---

## 🔌 API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Vérification état du serveur |
| `POST` | `/upload` | Upload d'un fichier CSV |
| `POST` | `/train` | Lancer l'entraînement ML |
| `GET` | `/experiments` | Liste des expériences MLflow |
| `GET` | `/experiments/{name}/runs` | Runs d'une expérience |
| `GET` | `/models` | Modèles dans le registry MLflow |
| `GET` | `/stats` | Statistiques globales |

### Exemple d'appel `/train`

```json
POST http://localhost:8000/train
{
  "dataset_id": "ds_1234567890",
  "task_type": "classification",
  "target": "Outcome",
  "features": ["Glucose", "BMI", "Age", "Insulin"],
  "algorithms": ["rf", "svm", "lr"],
  "test_size": 0.2,
  "experiment_name": "DiabetesML_v1"
}
```

---

## 🤖 Algorithmes disponibles

### Classification

| ID | Algorithme | Points forts |
|----|-----------|-------------|
| `rf` | Random Forest | Robuste, gère les données mixtes |
| `svm` | SVM | Efficace en haute dimension |
| `lr` | Logistic Regression | Simple, interprétable, rapide |
| `knn` | KNN | Basé sur la similarité |
| `dt` | Decision Tree | Très interprétable |
| `nb` | Naive Bayes | Rapide, bon sur texte |

### Régression

| ID | Algorithme | Points forts |
|----|-----------|-------------|
| `rf` | Random Forest | Robuste, gère les outliers |
| `lr` | Linear Regression | Simple et interprétable |
| `ridge` | Ridge | Évite l'overfitting (L2) |
| `lasso` | Lasso | Sélection de features (L1) |
| `svr` | SVR | Support Vector Regression |
| `dt` | Decision Tree | Arbre de régression |

---

## 📖 Utilisation

### Étape 1 — Préparer index.js

Dans `automl-frontend/src/index.js`, s'assurer que le contenu est :

```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

### Étape 2 — Choisir la bonne colonne cible

| Type de colonne | Tâche recommandée | Exemple |
|----------------|------------------|---------|
| 0/1, Oui/Non, catégories | Classification | `Outcome`, `activity`, `label` |
| Nombre continu | Régression | `price`, `score`, `temperature` |
| ID, nom unique | ❌ À éviter | `username`, `id`, `email` |

### Étape 3 — Réactiver l'env à chaque session

```powershell
# À faire à chaque ouverture de terminal
cd automl-backend
venv\Scripts\activate
```

---

## 🛠️ Dépannage

### ❌ `UNIQUE constraint failed` (MLflow)
**Cause** : L'expérience MLflow existe déjà.
**Solution** : Déjà corrigé dans `main.py` avec `get_or_create_experiment()`.

### ❌ `got 1 class` (SVM)
**Cause** : La colonne cible n'a qu'une seule valeur dans le train set.
**Solution** : Choisir une colonne avec au moins 2 valeurs distinctes. L'interface affiche maintenant un avertissement.

### ❌ `numpy==2.0.0rc1` incompatible
**Cause** : Python 3.14 incompatible.
**Solution** : Utiliser Python 3.11 avec l'environnement virtuel.

### ❌ `package.json not found`
**Cause** : `npm install` lancé dans `ml-generic/` au lieu de `automl-frontend/`.
**Solution** :
```powershell
npx create-react-app automl-frontend
xcopy /E /Y "ml-generic\src\*" "automl-frontend\src\"
cd automl-frontend && npm start
```

### ❌ Backend déconnecté (bandeau jaune)
**Cause** : FastAPI n'est pas démarré.
**Solution** :
```powershell
cd automl-backend
venv\Scripts\activate
uvicorn main:app --reload
```

### ❌ `Failed to load resource 400`
**Cause** : Mauvaise colonne cible choisie (une seule valeur unique).
**Solution** : Retourner à la configuration et choisir une autre colonne cible.

---

## 📦 Dépendances principales

### Back-End Python
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
mlflow>=2.10.0
joblib>=1.3.0
python-multipart==0.0.9
```

### Front-End React
```
react 18
react-dom 18
```

---

## 👩‍💻 Développé dans le cadre du module

**Machine Learning Avancée** — ING4 DS
Professeure : Aroua Hedhili

---

*AutoML Studio v1.0 — Application professionnelle ML avec React + FastAPI + MLflow*
