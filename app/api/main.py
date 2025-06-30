import io
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import pytz
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import AsyncMongoClient
from workers.celery_conn import celery
from workers.tasks import run_clustering_job

app = FastAPI()
BUCKET_NAME = "caas-data"
# Define the timezone
TIMEZONE = pytz.timezone("UTC")


def validate_data(data):
    # Check for None
    if data is None:
        raise ValueError("Input data is None.")

    # Check shape attribute
    if not hasattr(data, "shape"):
        raise TypeError("Input data must be array-like with a shape attribute.")

    # Check empty
    if data.size == 0:
        raise ValueError("Input data is empty.")

    # Check 2D shape
    if len(data.shape) != 2:
        raise ValueError("Input data must be 2-dimensional (samples, features).")

    # Check if numeric
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Input data must be numeric.")


async def get_mongodb():
    MONGODB_USER = os.getenv("MONGODB_USER")
    MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
    MONGODB_DB = os.getenv("MONGODB_DB")
    MONGODB_HOST = os.getenv("MONGODB_HOST")
    MONGODB_PORT = os.getenv("MONGODB_PORT")

    # For testing purposes, you can set MONGODB_URL in your environment variables
    MONGODB_URL = os.getenv("MONGODB_URL", None)

    if MONGODB_URL:
        mongodb_client = AsyncMongoClient(MONGODB_URL)
        print(f"Using MongoDB URL: {MONGODB_URL}")
    else:
        mongodb_client = AsyncMongoClient(
            f"mongodb://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}"
        )

    mongodb_database = mongodb_client.get_database(MONGODB_DB)
    try:
        yield mongodb_database
    finally:
        await mongodb_client.close()


class JobRequest(BaseModel):
    dataset_name: str
    columns: list[str]
    clustering_algorithm: str
    preprocess: bool = True
    user_id: str
    params: dict[str, Any] = {}  # Algorithm-specific params (e.g., n_clusters, eps)


class DatasetRequest(BaseModel):
    dataset_name: str
    user_id: str


class ResultPutRequest(BaseModel):
    job_id: str
    dataset_name: str
    columns: list[str]
    created_timestamp: str
    started_timestamp: str
    finished_timestamp: str
    clustering_algorithm: str
    params: dict[str, Any]
    labels: list[int]
    additional_results: dict[str, Any]
    user_id: str


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
        "params": req.params,
    }

    return response


@app.get("/dataset/{dataset_name}", response_class=StreamingResponse)
async def get_dataset(dataset_name: str, mongodb_database=Depends(get_mongodb)):
    try:
        # Debugging: Logge den übergebenen dataset_name
        print(f"Retrieving dataset: {dataset_name}")

        # Hole den Datensatz aus MongoDB
        data_collection = mongodb_database.get_collection("data")
        dataset = await data_collection.find_one(
            {"dataset_name": dataset_name}, {"_id": 0, "data": 1}
        )

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Debugging: Logge die abgerufenen Daten
        print(f"Dataset retrieved: {dataset}")

        # Erstelle einen StreamingResponse für die CSV-Daten
        file_stream = io.StringIO(dataset["data"])
        return StreamingResponse(
            file_stream,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_name}"},
        )
    except Exception as e:
        print(f"Error retrieving dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.put("/dataset/")
async def put_dataset(
    file: UploadFile = File(...),  # TODO: streaming file
    columns: list[str] = Form(...),
    clustering_algorithm: str = Form(...),
    preprocess: bool = Form(True),
    user_id: str = Form(...),
    params: str = Form("{}"),
    mongodb_database=Depends(get_mongodb),
):
    try:
        # Lese die Datei und extrahiere die Spaltennamen
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
            columns = df.columns.tolist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

        # Speichere den Datensatz und die Metadaten in MongoDB
        data_collection = mongodb_database.get_collection("data")
        await data_collection.insert_one(
            {
                "dataset_name": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "user_id": user_id,
                "columns": columns,
                "data": content.decode("utf-8"),  # Speichere die CSV-Daten als String
            }
        )

        # Starte den Clustering-Job und leite ihn an Celery weiter
        created_timestamp = datetime.now(TIMEZONE).isoformat()
        params_dict = json.loads(params) if isinstance(params, str) else params

        job = run_clustering_job.delay(
            file.filename,
            columns,
            created_timestamp,
            clustering_algorithm.lower(),
            preprocess,
            user_id,
            **params_dict,
        )

        return {
            "dataset_name": file.filename,
            "job_id": job.id,  # Dies ist die Celery-Job-ID
            "columns": columns,
            "clustering_algorithm": clustering_algorithm,
            "preprocess": preprocess,
            "user_id": user_id,
            "params": params_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.post("/result/")
async def post_result(req: ResultPutRequest, mongodb_database=Depends(get_mongodb)):
    try:
        result_collection = mongodb_database.get_collection("results")
        # Debugging: Logge die zu speichernden Daten
        print(f"Saving result: {req.model_dump()}")
        await result_collection.insert_one(req.model_dump())
        return {"job_id": req.job_id}
    except Exception as e:
        print(f"Error saving result: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving result: {e}") from e


@app.put("/upload/")
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    mongodb_database=Depends(get_mongodb),
):
    data_collection = mongodb_database.get_collection("data")
    # Check if file with the same name already exists
    exists = await data_collection.find_one({"dataset_name": file.filename})
    if exists:
        raise HTTPException(status_code=409, detail="Dataset with this name already exists.")
    try:
        content = await file.read()
        
        # Parse the CSV file to extract column names
        try:
            df = pd.read_csv(io.BytesIO(content))
            columns = df.columns.tolist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

        # Convert to Numpy array and validate
        data_array = df.to_numpy()
        validate_data(data_array)

        # Debugging: Logge die zu speichernden Daten
        print(f"Saving dataset: {file.filename}, columns: {columns}")

        # Store the dataset and metadata in MongoDB
        await data_collection.insert_one(
            {
                "dataset_name": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "user_id": user_id,
                "columns": columns,
                "data": content.decode("utf-8"),  # Store the CSV content as a string
            }
        )

        return {"dataset_name": file.filename, "columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.post("/cluster/")
async def start_clustering(
    dataset_name: str = Form(...),
    columns: str = Form(...),
    clustering_algorithm: str = Form(...),
    # TODO: Optimize preprocessing
    preprocess: bool = Form(True),
    user_id: str = Form(...),
    params: str = Form("{}"),
):
    try:
        created_timestamp = datetime.now(TIMEZONE).isoformat()
        columns_list = json.loads(columns) if isinstance(columns, str) else columns
        params_dict = json.loads(params) if isinstance(params, str) else params

        # print(f"Starting clustering job: module={module_name}, class={class_name}")
        # TODO: Validate params
        job = run_clustering_job.delay(
            dataset_name,
            columns_list,
            created_timestamp,
            clustering_algorithm,
            preprocess,
            user_id,
            **params_dict,
        )

        return {
            "dataset_name": dataset_name,
            "job_id": job.id,
            "columns": columns_list,
            "clustering_algorithm": clustering_algorithm,
            "preprocess": preprocess,
            "user_id": user_id,
            "params": params_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting clustering: {str(e)}") from e


@app.get("/datasets/")
async def list_datasets(mongodb_database=Depends(get_mongodb)):
    """
    Returns a list of all uploaded datasets with their user IDs.
    """
    data_collection = mongodb_database.get_collection("data")
    datasets = await data_collection.find({}, {"_id": 0, "dataset_name": 1, "user_id": 1}).to_list(
        length=1000
    )
    return datasets


@app.delete("/datasets/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    mongodb_database=Depends(get_mongodb),
):
    """
    Löscht einen Datensatz aus MongoDB.
    """
    data_collection = mongodb_database.get_collection("data")
    result = await data_collection.delete_one({"dataset_name": dataset_name})

    if result.deleted_count == 1:
        return {"detail": "Dataset deleted"}
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")


@app.get("/debug/job/{job_id}")
async def debug_job(job_id: str, mongodb_database=Depends(get_mongodb)):
    """
    Debug endpoint to check job status and results
    """
    try:
        # Check Celery task
        task = celery.AsyncResult(job_id)
        task_info = {
            "task_id": task.id,
            "status": task.status,
            "result": task.result if task.ready() else None,
        }

        # Check MongoDB
        results_collection = mongodb_database.get_collection("results")
        stored_result = await results_collection.find_one({"job_id": job_id})

        return {"task_info": task_info, "stored_result": stored_result is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}") from e


@app.post("/automl/cluster")
def start_automl_dummy():
    task = celery.send_task("automl_worker.hello_automl")
    return JSONResponse(content={"task_id": task.id})


@app.get("/cluster/{task_id}/table")
async def get_clustering_result_table(
    task_id: str,
    mongodb_database=Depends(get_mongodb),
):
    try:
        result_collection = mongodb_database.get_collection("results")
        result = await result_collection.find_one({"job_id": task_id})
        if not result:
            raise HTTPException(
                status_code=404, detail=f"Result not found for given job_id: {task_id}"
            )

        additional = result.get("additional_results", {})
        labels = result.get("labels")
        X = additional.get("X")
        columns = additional.get("columns")

        if X is not None and columns is not None:
            df = []
            for row, label in zip(X, labels, strict=False):
                row_dict = {col: val for col, val in zip(columns, row, strict=False)}
                row_dict["Cluster"] = label
                df.append(row_dict)
            return {"data": df, "columns": columns + ["Cluster"]}
        else:
            return {"labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.get("/cluster/{task_id}/raw")
async def get_clustering_result_raw(
    task_id: str,
    mongodb_database=Depends(get_mongodb),
):
    try:
        result_collection = mongodb_database.get_collection("results")
        result = await result_collection.find_one({"job_id": task_id})
        if not result:
            raise HTTPException(
                status_code=404, detail=f"Result not found for given job_id: {task_id}"
            )
        return result.get("labels")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.get("/cluster/{task_id}/graph")
async def get_clustering_result_graph(
    task_id: str,
    mongodb_database=Depends(get_mongodb),
):
    try:
        result_collection = mongodb_database.get_collection("results")
        result = await result_collection.find_one({"job_id": task_id})
        if not result:
            raise HTTPException(
                status_code=404, detail=f"Result not found for given job_id: {task_id}"
            )

        additional = result.get("additional_results", {})
        labels = result.get("labels")
        X = additional.get("X")
        columns = additional.get("columns")

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
            title=f"Clustering: {result.get('clustering_algorithm')}",
        )
        return fig.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.get("/jobs/")
async def list_jobs(mongodb_database=Depends(get_mongodb)):
    """
    Gibt eine Übersicht aller bekannten Jobs inkl. Status zurück.
    """
    try:
        results_collection = mongodb_database.get_collection("results")
        jobs = await results_collection.find({}, {"_id": 0}).to_list(length=1000)
        job_list = []
        for job in jobs:
            job_id = job.get("job_id")
            celery_status = None
            if job_id:
                task = celery.AsyncResult(job_id)
                celery_status = task.status
            job_list.append(
                {
                    "job_id": job_id,
                    "dataset_name": job.get("dataset_name"),
                    "created_timestamp": job.get("created_timestamp"),
                    "clustering_algorithm": job.get("clustering_algorithm"),
                    "user_id": job.get("user_id"),
                    "status": celery_status,
                }
            )
        return job_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {e}") from e
