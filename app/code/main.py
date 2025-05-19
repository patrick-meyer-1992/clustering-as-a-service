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
    dataset_url: str
    columns: List[str]
    clustering_algorithm: str  
    user_id: str
    params: Dict[str, Any] = {}  # Algorithm-specific params (e.g., n_clusters, eps)

class DatasetRequest(BaseModel):
    dataset_url: str
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

    job = run_clustering_job.delay(
        req.dataset_url,
        req.columns, 
        req.clustering_algorithm.lower(), 
        req.user_id,
        req.params
        )
    
    response = {
        "job_id": job.id,
        "dataset_url": req.dataset_url,
        "columns": req.columns,
        "created_timestamp": datetime.now(TIMEZONE).isoformat(),
        "clustering_algorithm": req.clustering_algorithm,
        "user_id": req.user_id,
        "params": req.params
    }

    return response

@app.get("/dataset/{filename}", response_class=StreamingResponse)
async def get_dataset(filename: str, background_tasks: BackgroundTasks):
    try:
        minio_response = minio_client.get_object(BUCKET_NAME, filename)
        background_tasks.add_task(minio_response.close)
        background_tasks.add_task(minio_response.release_conn)
        return StreamingResponse(
            minio_response,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
            background=background_tasks
        )

    except S3Error as err:
        raise HTTPException(status_code=404, detail=f"MinIO error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



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


@app.put("/dataset/")
async def put_dataset(file: UploadFile = File(...)):
    # TODO:
    # Check if the file is a valid CSV 
    # Check if file already exists in MinIO
    # Check if the columns are valid

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
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
        })

        return {"filename": file.filename}

    except S3Error as err:
        raise HTTPException(status_code=500, detail=f"MinIO error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
