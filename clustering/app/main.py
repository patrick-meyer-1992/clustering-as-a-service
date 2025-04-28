from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from pymongo import MongoClient
import numpy as np
from datetime import datetime

# MongoDB-Verbindung aufbauen
client = MongoClient("mongodb://localhost:27017/")
db = client["clustering_db"]
requests_collection = db["cluster_requests"]
results_collection = db["cluster_results"]

app = FastAPI(
    title="Clustering API",
    description="Eine API zum Clustern von Punkten mit verschiedenen Algorithmen und Speicherung in MongoDB.",
    version="1.0.0"
)

class ClusterRequest(BaseModel):
    points: List[List[float]]
    algorithm: str = "kmeans"              # "kmeans", "dbscan", "agglomerative"
    n_clusters: Optional[int] = 3          # Für KMeans / Agglomerative
    eps: Optional[float] = 0.5             # Für DBSCAN
    min_samples: Optional[int] = 5         # Für DBSCAN

class ClusterResponse(BaseModel):
    labels: List[int]
    centers: Optional[List[List[float]]] = None   # Nur für einige Algorithmen

@app.post("/cluster", response_model=ClusterResponse)
def cluster(data: ClusterRequest):
    try:
        points = np.array(data.points)

        # Anfrage in die MongoDB speichern
        request_record = {
            "points": data.points,
            "algorithm": data.algorithm,
            "n_clusters": data.n_clusters,
            "eps": data.eps,
            "min_samples": data.min_samples,
            "timestamp": datetime.utcnow()
        }
        request_insert_result = requests_collection.insert_one(request_record)
        request_id = str(request_insert_result.inserted_id)

        # Clustering durchführen
        algorithm = data.algorithm.lower()
        if algorithm == "kmeans":
            model = KMeans(n_clusters=data.n_clusters)
            model.fit(points)
            labels = model.labels_.tolist()
            centers = model.cluster_centers_.tolist()
        elif algorithm == "dbscan":
            model = DBSCAN(eps=data.eps, min_samples=data.min_samples)
            model.fit(points)
            labels = model.labels_.tolist()
            centers = None
        elif algorithm == "agglomerative":
            model = AgglomerativeClustering(n_clusters=data.n_clusters)
            model.fit(points)
            labels = model.labels_.tolist()
            centers = None
        else:
            raise ValueError("Unsupported algorithm. Choose 'kmeans', 'dbscan', or 'agglomerative'.")

        # Ergebnis in die MongoDB speichern
        result_record = {
            "request_id": request_id,
            "labels": labels,
            "centers": centers,
            "timestamp": datetime.utcnow()
        }
        results_collection.insert_one(result_record)

        # Antwort zurückgeben
        return ClusterResponse(
            labels=labels,
            centers=centers
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from bson import ObjectId  # Um MongoDB-IDs korrekt zu verarbeiten

@app.get("/results/{request_id}", response_model=ClusterResponse)
def get_result(request_id: str):
    try:
        # Suche das Ergebnis in der MongoDB anhand der request_id
        result = results_collection.find_one({"request_id": request_id})

        if result is None:
            # Wenn kein Ergebnis gefunden wurde, gebe einen 404-Fehler zurück
            raise HTTPException(status_code=404, detail="Ergebnis nicht gefunden.")

        # Rückgabe des gefundenen Ergebnisses im ClusterResponse-Format
        return ClusterResponse(
            labels=result["labels"],
            centers=result["centers"]
        )

    except Exception as e:
        # Fehlerbehandlung, z.B. ungültige ID oder Verbindungsprobleme
        raise HTTPException(status_code=400, detail=str(e))
