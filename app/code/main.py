from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from tasks import run_clustering_task
from celery_conn import celery
import numpy as np
import os

app = FastAPI()

class ClusteringRequest(BaseModel):
    data: List[List[float]]
    algorithm: str  # "kmeans" or "dbscan"
    params: Dict[str, Any] = {}  # Algorithm-specific params (e.g., n_clusters, eps)

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/cluster/")
def submit_clustering_job(req: ClusteringRequest):
    task = run_clustering_task.delay(req.data, req.algorithm.lower(), req.params)
    return {"task_id": task.id}

@app.get("/cluster/{task_id}")
def get_clustering_result(task_id: str):
    pass
    # res = celery.AsyncResult(task_id)

    # if res.state == "PENDING":
    #     return {"status": "pending"}

    # if res.state == "FAILURE":
    #     return {"status": "failed", "error": str(res.result)}

    # if res.state == "SUCCESS":
    #     job_id = res.result.get("job_id")
    #     label_path = res.result.get("labels_path")

    #     if not label_path or not os.path.exists(label_path):
    #         raise HTTPException(status_code=404, detail="Labels not found")

    #     labels = np.load(label_path).tolist()
    #     return {
    #         "status": "completed",
    #         "algorithm": res.result.get("algorithm"),
    #         "job_id": job_id,
    #         "labels": labels
    #     }

    # return {"status": res.state}
