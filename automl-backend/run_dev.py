#!/usr/bin/env python
"""
Démarre l'API en mode développement.

Important : sans --reload-exclude, uvicorn redémarre à chaque écriture MLflow
dans mlruns/ → la requête POST /train est coupée → « Failed to fetch » / CORS.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=[
            "mlruns",
            "mlruns/*",
            "datasets",
            "datasets/*",
            "data",
            "data/*",
            "*.db",
            "*.db-wal",
            "*.db-shm",
            "__pycache__",
            "__pycache__/*",
        ],
    )
