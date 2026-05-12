# AutoML Studio — Application Professionnelle
## React + FastAPI + MLflow

---

## Architecture

```
automl-studio/
├── automl-backend/         ← Back-End Python
│   ├── main.py             ← FastAPI + MLflow
│   ├── requirements.txt
│   └── mlflow.db           ← Créé automatiquement
│
└── automl-frontend/        ← Front-End React
    └── src/
        ├── App.jsx
        ├── App.css
        ├── api.js          ← Connexion FastAPI
        └── components/
            ├── UploadStep.jsx
            ├── ConfigStep.jsx
            ├── Dashboard.jsx
            └── MLflowPanel.jsx
```

---

## Installation

### 1 — Back-End (FastAPI + MLflow)

```bash
cd automl-backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2 — MLflow UI (dans un autre terminal)

```bash
cd automl-backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 3 — Front-End (React)

```bash
npx create-react-app automl-frontend
cd automl-frontend
# Copier le contenu du dossier src/ dans automl-frontend/src/
npm start
```

---

## URLs

| Service     | URL                        |
|-------------|---------------------------|
| React App   | http://localhost:3000      |
| FastAPI     | http://localhost:8000      |
| API Docs    | http://localhost:8000/docs |
| MLflow UI   | http://localhost:5000      |

---

## Fonctionnalités

- Upload CSV drag & drop
- Auto-détection Classification / Régression
- 6 algorithmes Classification : RF, SVM, LR, KNN, DT, NB
- 6 algorithmes Régression    : RF, LR, Ridge, Lasso, SVR, DT
- MLflow tracking automatique de tous les runs
- Model Registry MLflow
- Feature Importance
- Matrice de confusion
- Analyse Overfitting (Train vs Test)
- Export CSV des résultats
- Dashboard MLflow intégré dans l'interface
