# AutoML Studio — Back-end

API **FastAPI** + **MLflow** + génération de rapports.

Documentation complète (installation, API, architecture) : **[README à la racine du projet](../README.md)**.

## Démarrage rapide

```powershell
cd automl-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Swagger : http://localhost:8000/docs
