from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from tasks import run_clustering_job
from celery_conn import celery
import os
from minio import Minio
from minio.error import S3Error
import io
from datetime import datetime
import pytz
from pymongo import AsyncMongoClient
from fastapi import Query
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import uuid
from fastapi import Form
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse

app = FastAPI()

BUCKET_NAME = "caas-data"

MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_HOST = os.getenv("MINIO_HOST")
MINIO_PORT = os.getenv("MINIO_PORT")

# Configure MinIO client
minio_client = Minio(
    endpoint=f"{MINIO_HOST}:{MINIO_PORT}",  
    access_key=MINIO_ACCESS_KEY,   
    secret_key=MINIO_SECRET_KEY,    
    secure=False                
)
# Ensure the bucket exists
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_HOST = os.getenv("MONGODB_HOST")
MONGODB_PORT = os.getenv("MONGODB_PORT")

mongodb_client = AsyncMongoClient(f"mongodb://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}")
mongodb_database = mongodb_client.get_database(MONGODB_DB)

# Define the timezone
TIMEZONE = pytz.timezone("UTC")
class JobRequest(BaseModel):
    dataset_name: str
    columns: List[str]
    clustering_algorithm: str
    preprocess: bool = True
    user_id: str
    params: Dict[str, Any] = {}  # Algorithm-specific params (e.g., n_clusters, eps)

class DatasetRequest(BaseModel):
    dataset_name: str
    user_id: str

class ResultPutRequest(BaseModel):
    job_id: str
    dataset_name: str
    columns: List[str]
    created_timestamp: str
    started_timestamp: str
    finished_timestamp: str
    clustering_algorithm: str
    params: Dict[str, Any]
    labels: List[int]
    additional_results: Dict[str, Any]
    user_id: str

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/job/")
def post_job(req: JobRequest):
    # TODO:
    # Validate the request
    # Check if the dataset URL is valid and dataset exists
    # Check if the columns are valid
    # Check if the clustering algorithm is supported
    # Check if the user ID is valid
    # Check if the params are valid

    created_timestamp = datetime.now(TIMEZONE).isoformat()

    job = run_clustering_job.delay(
        req.dataset_name,
        req.columns, 
        created_timestamp,
        req.clustering_algorithm.lower(), 
        req.preprocess,
        req.user_id,
        **req.params,
        )
    
    response = {
        "job_id": job.id,
        "dataset_name": req.dataset_name,
        "columns": req.columns,
        "created_timestamp": created_timestamp,
        "clustering_algorithm": req.clustering_algorithm.lower(),
        "preprocess": req.preprocess,
        "user_id": req.user_id,
        "params": req.params
    }

    return response

@app.get("/dataset/{dataset_name}", response_class=StreamingResponse)
async def get_dataset(dataset_name: str, background_tasks: BackgroundTasks):
    try:
        minio_response = minio_client.get_object(BUCKET_NAME, dataset_name)
        background_tasks.add_task(minio_response.close)
        background_tasks.add_task(minio_response.release_conn)
        return StreamingResponse(
            minio_response,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_name}"},
            background=background_tasks
        )

    except S3Error as err:
        raise HTTPException(status_code=404, detail=f"MinIO error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# Get clustering result by task_id | Forwarding results to the frontend
@app.get("/cluster/{task_id}")
async def get_clustering_result(
    task_id: str,
    presentation: str = Query("table", enum=["table", "raw", "graph"])
):
    
    # get result from MongoDB
    result_collection = mongodb_database.get_collection("results")
    result = await result_collection.find_one({"job_id": task_id})

    # Check if result exists
    if not result:
        raise HTTPException(status_code=404, detail=f"Result not found for given job_id: {task_id}")

    # get metadata and additional results
    additional = result.get("additional_results", {})
    labels = result.get("labels")
    X = additional.get("X")
    columns = additional.get("columns")

    # table presentation
    if presentation == "table":
        if X is not None and columns is not None:
            df = []
            for row, label in zip(X, labels):
                row_dict = {col: val for col, val in zip(columns, row)}
                row_dict["Cluster"] = label
                df.append(row_dict)
            return {"data": df, "columns": columns + ["Cluster"]}
        else:
            return {"labels": labels}

    # raw presentation
    if presentation == "raw":
        return labels

    # graph presentation
    if presentation == "graph":
        if X is None or labels is None or len(X) == 0 or len(labels) == 0:
            raise HTTPException(status_code=400, detail="No data for plotting")
        X_np = np.array(X)
        labels_np = np.array(labels)
        if X_np.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Data is not 2D")
        fig = px.scatter(
            x=X_np[:, 0],
            y=X_np[:, 1],
            color=labels_np.astype(str),
            title=f"Clustering: {result.get('clustering_algorithm')}"
        )
        return fig.to_dict()

    raise HTTPException(status_code=400, detail="Invalid presentation type")
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


@app.put("/dataset/")
async def put_dataset(
    file: UploadFile = File(...), # TODO: streaming file
    columns: List[str] = Form(...),
    clustering_algorithm: str = Form(...),
    preprocess: bool = Form(True),
    user_id: str = Form(...),
    params: str = Form("{}")
):

    # After data upload, the file is stored in MinIO and a clustering job is started.

    try:
        # Read file content
        content = await file.read()
        file_stream = io.BytesIO(content)

        # Upload to MinIO
        minio_client.put_object(
            BUCKET_NAME,
            file.filename,
            data=file_stream,
            length=len(content),
            content_type=file.content_type
        )

        data_collection = mongodb_database.get_collection("data")
        await data_collection.insert_one({
            "dataset_name": file.filename,
            "content_type": file.content_type,
            "size": len(content),
        })

        # start Clustering-Job and forward it to Celery
        created_timestamp = datetime.now(TIMEZONE).isoformat()
        params_dict = json.loads(params) if isinstance(params, str) else params

        # run_clustering_job is a Celery task that processes the clustering job, returning a job ID.
        job = run_clustering_job.delay(
            file.filename,
            columns,
            created_timestamp,
            clustering_algorithm.lower(),
            preprocess,
            user_id,
            **params_dict
        )

        return {
            "dataset_name": file.filename,
            "job_id": job.id, # This is the Celery job ID
            "columns": columns,
            "clustering_algorithm": clustering_algorithm,
            "preprocess": preprocess,
            "user_id": user_id,
            "params": params_dict
        }

    except S3Error as err:
        raise HTTPException(status_code=500, detail=f"MinIO error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
@app.post("/result/")
async def post_result(req: ResultPutRequest):
    result_collection = mongodb_database.get_collection("results")
    await result_collection.insert_one(req.model_dump())
    return {"job_id": req.job_id}

@app.put("/upload/")
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    data_collection = mongodb_database.get_collection("data")
    # Check if file with the same name already exists
    exists = await data_collection.find_one({"dataset_name": file.filename})
    if exists:
        raise HTTPException(status_code=409, detail="Dataset with this name already exists.")
    try:
        content = await file.read()
        file_stream = io.BytesIO(content)

        # Upload to MinIO
        minio_client.put_object(
            BUCKET_NAME,
            file.filename,
            data=file_stream,
            length=len(content),
            content_type=file.content_type
        )

        await data_collection.insert_one({
            "dataset_name": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "user_id": user_id
        })

        return {"dataset_name": file.filename}
    except S3Error as err:
        raise HTTPException(status_code=500, detail=f"MinIO error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/cluster/")
async def start_clustering(
    dataset_name: str = Form(...),
    columns: str = Form(...),  # JSON-String
    clustering_algorithm: str = Form(...),
    preprocess: bool = Form(True),
    user_id: str = Form(...),
    params: str = Form("{}")
):
    try:
        created_timestamp = datetime.now(TIMEZONE).isoformat()
        columns_list = json.loads(columns) if isinstance(columns, str) else columns
        params_dict = json.loads(params) if isinstance(params, str) else params

        job = run_clustering_job.delay(
            dataset_name,
            columns_list,
            created_timestamp,
            clustering_algorithm.lower(),
            preprocess,
            user_id,
            **params_dict
        )

        return {
            "dataset_name": dataset_name,
            "job_id": job.id,
            "columns": columns_list,
            "clustering_algorithm": clustering_algorithm,
            "preprocess": preprocess,
            "user_id": user_id,
            "params": params_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.get("/datasets/")
async def list_datasets():
    """
    Returns a list of all uploaded datasets with their user IDs.
    """
    data_collection = mongodb_database.get_collection("data")
    datasets = await data_collection.find(
        {}, 
        {"_id": 0, "dataset_name": 1, "user_id": 1}
    ).to_list(length=1000)
    return datasets  # Returns list of dicts with dataset_name and user_id

@app.delete("/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """
    Deletes a dataset from MongoDB and MinIO.
    """
    data_collection = mongodb_database.get_collection("data")
    result = await data_collection.delete_one({"dataset_name": dataset_name})
    try:
        minio_client.remove_object(BUCKET_NAME, dataset_name)
    except Exception:
        pass  # Ignore if file does not exist in MinIO
    if result.deleted_count == 1:
        return {"detail": "Dataset deleted"}
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")