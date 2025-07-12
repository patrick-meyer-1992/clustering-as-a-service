import io
import os
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
import plotly.express as px
import pytz
from clustering import wrappers
from clustering.preprocessing_params import PreProcessingParams
from fastapi import Depends, FastAPI, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pymongo import AsyncMongoClient
from workers.celery_conn import celery
from workers.tasks import run_clustering_job

app = FastAPI()

# Define the timezone
TIMEZONE = pytz.timezone("UTC")

ALGORITHM_MAP = {
    getattr(wrappers, algo).backend_name: getattr(wrappers, algo) for algo in dir(wrappers) if algo.endswith("Wrapper")
}
algorithms = [backend_name for backend_name in ALGORITHM_MAP]


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
    # if not np.issubdtype(data.dtype, np.number):
    #    raise TypeError("Input data must be numeric.")


async def get_mongodb():
    MONGODB_DB = os.getenv("MONGODB_DB")
    MONGODB_HOST = os.getenv("MONGODB_HOST")
    MONGODB_PORT = os.getenv("MONGODB_PORT")

    # For testing purposes, you can set MONGODB_URL in your environment variables
    MONGODB_URL = os.getenv("MONGODB_URL", None)

    if MONGODB_URL:
        mongodb_client = AsyncMongoClient(MONGODB_URL)
        print(f"Using MongoDB URL: {MONGODB_URL}")
    else:
        mongodb_client = AsyncMongoClient(f"mongodb://{MONGODB_HOST}:{MONGODB_PORT}")

    mongodb_database = mongodb_client.get_database(MONGODB_DB)
    try:
        yield mongodb_database
    finally:
        await mongodb_client.close()


class DatasetField(str, Enum):
    dataset_name = "dataset_name"
    content_type = "content_type"
    columns = "columns"
    size = "size"


@app.get(
    "/metadata/{dataset_name}",
    response_model=dict[str, str | list[dict[str, str | list[str]]]],
)
async def get_metadata(
    dataset_name: str,
    fields: list[DatasetField] = Query(
        [DatasetField.columns],
        description="Fields to return from the dataset metadata. Default is 'columns'.",
    ),
    mongodb_database=Depends(get_mongodb),
):
    try:
        data_collection = mongodb_database.get_collection("data")
        # Check if the dataset exists
        exists = await data_collection.find_one({"dataset_name": dataset_name})
        if not exists:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Fetch the specified fields from the dataset
        fields_to_return = {field.value: 1 for field in fields}
        fields_to_return["_id"] = 0  # Exclude the _id field
        response = await data_collection.find_one({"dataset_name": dataset_name}, fields_to_return)

        return response
    except Exception as e:
        print(f"Error retrieving dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.get("/dataset/{dataset_name}", response_class=StreamingResponse)
async def get_dataset(
    dataset_name: str,
    mongodb_database=Depends(get_mongodb),
):
    try:
        # Hole den Datensatz aus MongoDB
        data_collection = mongodb_database.get_collection("data")
        dataset = await data_collection.find_one({"dataset_name": dataset_name}, {"_id": 0, "data": 1})

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

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


class DatasetPutResponse(BaseModel):
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[dict[str, str | list[str]]] = Field(
        description="The columns used in the dataset and their allowed types",
        default=...,
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "columns": [
                        {
                            "name": "sepal.length",
                            "allowed_types": ["numeric", "nominal", "ordinal"],
                        },
                        {
                            "name": "sepal.width",
                            "allowed_types": ["numeric", "nominal", "ordinal"],
                        },
                        {
                            "name": "petal.length",
                            "allowed_types": ["numeric", "nominal", "ordinal"],
                        },
                        {
                            "name": "petal.width",
                            "allowed_types": ["numeric", "nominal", "ordinal"],
                        },
                    ],
                }
            ]
        }
    }


@app.put("/dataset/", response_model=DatasetPutResponse)
async def put_dataset(
    file: UploadFile = File(description="The CSV file to upload", default=...),
    mongodb_database=Depends(get_mongodb),
):
    type_mapping = {
        "float64": ["numeric", "nominal", "ordinal"],
        "float32": ["numeric", "nominal", "ordinal"],
        "int64": ["numeric", "nominal", "ordinal"],
        "int32": ["numeric", "nominal", "ordinal"],
        "bool": ["ordinal", "nominal"],
        "object": ["nominal", "ordinal"],
        "category": ["nominal", "ordinal"],
        "datetime64[ns]": None,
        "timedelta[ns]": None,
    }

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
            columns = [
                {"name": col, "allowed_types": type_mapping.get(str(dtype))}
                for col, dtype in df.dtypes.items()
                if type_mapping[str(dtype)] is not None
            ]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}") from e

        # Convert to Numpy array and validate
        data_array = df.to_numpy()
        validate_data(data_array)

        # Store the dataset and metadata in MongoDB
        await data_collection.insert_one(
            {
                "dataset_name": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                "columns": columns,
                "data": content.decode("utf-8"),  # Store the CSV content as a string
            }
        )

        return {"dataset_name": file.filename, "columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


class DatasetDeleteResponse(BaseModel):
    dataset_name: str = Field(description="The name of the deleted dataset", default=...)
    job_ids: list[str] | None = Field(description="List of job IDs associated with the deleted dataset", default=None)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "job_ids": [
                        "fb231936-1d83-43de-85a4-81c6889dd21c",
                        "777b6e2e-07b6-47a8-82e1-f08900ea0176",
                    ],
                }
            ]
        }
    }


@app.delete("/dataset/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    mongodb_database=Depends(get_mongodb),
) -> DatasetDeleteResponse:
    """
    Delete a dataset from mongoDB and all results associated with it.
    """
    data_collection = mongodb_database.get_collection("data")
    result_collection = mongodb_database.get_collection("results")
    # Get all job IDs associated with the dataset
    jobs = await result_collection.find({"dataset_name": dataset_name}, {"_id": 0, "job_id": 1}).to_list(length=None)
    await result_collection.delete_many({"dataset_name": dataset_name})
    result = await data_collection.delete_one({"dataset_name": dataset_name})

    if result.deleted_count == 1:
        return DatasetDeleteResponse(dataset_name=dataset_name, job_ids=[job["job_id"] for job in jobs])
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")


class DatasetGetResponse(BaseModel):
    dataset_name: str = Field(description="The name of the uploaded dataset", default=...)
    model_config = {"json_schema_extra": {"examples": [{"dataset_name": "iris.csv"}]}}


@app.get("/datasets/")
async def get_datasets(
    mongodb_database=Depends(get_mongodb),
) -> list[DatasetGetResponse]:
    """
    Returns a list of all uploaded datasets.
    """
    data_collection = mongodb_database.get_collection("data")
    datasets = await data_collection.find({}, {"_id": 0, "dataset_name": 1}).to_list(length=1000)
    return [DatasetGetResponse(**dataset) for dataset in datasets]


class JobPostRequest(BaseModel):
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[dict[str, str]] | None = Field(description="The columns to use for clustering", default=None)
    clustering_algorithm: str = Field(description="The clustering algorithm to use", default=..., examples=algorithms)
    preprocess: bool = Field(description="Whether to preprocess the data", default=True)
    clustering_params: dict[str, Any] | None = Field(description="Clustering algorithm parameters", default=None)
    preprocessing_params: PreProcessingParams | None = Field(description="Preprocessing parameters", default=None)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "columns": [
                        {
                            "name": "sepal.length",
                            "type": "numeric",
                        },
                        {
                            "name": "sepal.width",
                            "type": "numeric",
                        },
                    ],
                    "clustering_algorithm": "kmeans",
                    "preprocess": "true",
                    "clustering_params": {"n_clusters": 3},
                    "preprocessing_params": {
                        "scaler": "auto",
                        "use_normalization": False,
                        "normalization_type": "l2",
                        "use_pca": False,
                        "pca_components": 10,
                        "transform_type": "quantile",
                        "imputation_strategy": "mean",
                        "outlier_removal": "zscore",
                        "outlier_threshold": 3.0,
                        "feature_selection": "low_variance",
                        "variance_threshold": 0.0,
                    },
                }
            ]
        }
    }


class JobPostResponse(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    model_config = {"json_schema_extra": {"examples": [{"job_id": "fb231936-1d83-43de-85a4-81c6889dd21c"}]}}


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    for key, value in params.items():
        if not isinstance(value, str):
            continue
        if value.lower() == "inf" or value.lower() == "-inf":
            params[key] = float(value)
        elif value.lower() == "none":
            params[key] = None
        elif value.lower() in ["true", "false"]:
            params[key] = value.lower() == "true"
        elif value.isdigit():
            params[key] = int(value)
        elif value.replace(".", "", 1).isdigit():
            params[key] = float(value)
    return params


@app.post("/job/")
async def post_job(
    req: JobPostRequest,
    mongodb_database=Depends(get_mongodb),
) -> JobPostResponse:
    clustering_params = parse_params(req.clustering_params)

    data_collection = mongodb_database.get_collection("data")
    # Check if the dataset exists
    exists = await data_collection.find_one({"dataset_name": req.dataset_name})
    if not exists:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Fetch the specified fields from the dataset
    dataset = await data_collection.find_one({"dataset_name": req.dataset_name}, {"_id": 0, "columns": 1})

    # Check if requested columns exist for the dataset and their types are valid
    for request_column in req.columns:
        if request_column["name"] not in [col["name"] for col in dataset["columns"]]:
            print(request_column["name"])
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request_column['name']}' not found in dataset '{req.dataset_name}'",
            )
        allowed_types = [col["allowed_types"] for col in dataset["columns"] if col["name"] == request_column["name"]][0]

        if request_column["type"] not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request_column['name']}' does not support type '{request_column['type']}'.",
            )

    try:
        job = run_clustering_job.delay(
            req.dataset_name,
            req.columns,
            datetime.now(TIMEZONE).isoformat(),
            req.clustering_algorithm,
            req.preprocess,
            preprocessing_params=(req.preprocessing_params.model_dump() if req.preprocess else None),
            **(clustering_params),
        )

        return JobPostResponse(
            job_id=job.id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting clustering: {str(e)}") from e


class JobsGetResponse(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    dataset_name: str = Field(description="The name of the dataset", default=...)
    created_timestamp: str = Field(description="The creation timestamp", default=...)
    clustering_algorithm: str = Field(description="The clustering algorithm used", default=...)
    status: str | None = Field(description="The status of the job", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c",
                    "created_timestamp": "2025-07-02T11:53:56.083632+00:00",
                    "clustering_algorithm": "kmeans",
                    "status": "SUCCESS",
                }
            ]
        }
    }


@app.get("/jobs/")
async def get_jobs(mongodb_database=Depends(get_mongodb)) -> list[JobsGetResponse]:
    """
    Returns an overview of all known jobs including their status.
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
                JobsGetResponse(
                    job_id=job_id,
                    dataset_name=job.get("dataset_name"),
                    created_timestamp=job.get("created_timestamp"),
                    clustering_algorithm=job.get("clustering_algorithm"),
                    status=celery_status,
                )
            )
        return job_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {e}") from e


@app.get("/result/{job_id}/table")
async def get_result_table(
    job_id: str,
    mongodb_database=Depends(get_mongodb),
) -> dict[Any, Any]:
    try:
        result_collection = mongodb_database.get_collection("results")
        data_collection = mongodb_database.get_collection("data")
        result = await result_collection.find_one({"job_id": job_id})

        if not result:
            raise HTTPException(status_code=404, detail=f"Result not found for given job_id: {job_id}")

        dataset = await data_collection.find_one({"dataset_name": result.get("dataset_name")}, {"_id": 0, "data": 1})
        df = pd.read_csv(io.StringIO(dataset["data"]))
        df["labels"] = result.get("labels", [])

        return df.to_dict(orient="dict")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


class ResultField(str, Enum):
    job_id = "job_id"
    dataset_name = "dataset_name"
    columns = "columns"
    created_timestamp = "created_timestamp"
    started_timestamp = "started_timestamp"
    finished_timestamp = "finished_timestamp"
    clustering_algorithm = "clustering_algorithm"
    clustering_params = "clustering_params"
    preprocessing_params = "preprocessing_params"
    labels = "labels"
    additional_results = "additional_results"


@app.get("/result/{job_id}/raw")
async def get_result_raw(
    job_id: str,
    field: ResultField | None = Query(
        "labels",
        title="The field to return from the result. If not set, only the labels are returned.",
    ),
    mongodb_database=Depends(get_mongodb),
) -> Any:
    try:
        result_collection = mongodb_database.get_collection("results")
        result = await result_collection.find_one({"job_id": job_id}, {"_id": 0, field: 1})
        if not result:
            raise HTTPException(status_code=404, detail=f"Result not found for given job_id: {job_id}")
        return result.get(field)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


@app.get("/result/{job_id}/graph")
async def get_result_graph(
    job_id: str,
    x_column: str | None = Query(
        None,
        title="The column to use for the x-axis. Set only in combination with y_column",
    ),
    y_column: str | None = Query(
        None,
        title="The column to use for the y-axis. Set only in combination with x_column",
    ),
    mongodb_database=Depends(get_mongodb),
) -> dict[str, Any]:
    """
    Creates a 2D plotly scatter plot for the clustering result of a job.

    If x_column or y_column are not provided, the first two columns of the result will be used.
    """
    try:
        result_collection = mongodb_database.get_collection("results")
        data_collection = mongodb_database.get_collection("data")

        result = await result_collection.find_one({"job_id": job_id})
        if not result:
            raise HTTPException(status_code=404, detail=f"Result not found for given job_id: {job_id}")

        dataset = await data_collection.find_one({"dataset_name": result.get("dataset_name")}, {"_id": 0, "data": 1})
        df = pd.read_csv(io.StringIO(dataset["data"]))
        labels = [str(label) for label in result.get("labels")]
        if df is None or labels is None or df.shape[0] == 0 or df.shape[1] < 2 or len(labels) == 0:
            raise HTTPException(status_code=400, detail="No data for plotting")

        if x_column is None or y_column is None:
            x_column = result.get("columns")[0]
            y_column = result.get("columns")[1]
        elif x_column not in df.columns or y_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid x_column or y_column")

        df = df[[x_column, y_column]]

        if df.shape[1] != 2:
            raise HTTPException(status_code=400, detail="Data is not 2D")
        # Ensure legend items appear in a certain order by specifying category_orders
        unique_labels = sorted(set(labels))
        fig = px.scatter(
            x=df[x_column],
            y=df[y_column],
            color=labels,
            title=f"Clustering: {result.get('clustering_algorithm')}",
            labels={"x": x_column, "y": y_column},
            category_orders={"color": unique_labels},
        )
        return fig.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e


class ResultPostRequest(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    dataset_name: str = Field(description="The name of the dataset", default=...)
    columns: list[dict[str, str]] = Field(
        description="The columns used in the dataset and their used encoding",
        default=...,
    )
    created_timestamp: str = Field(description="The creation timestamp", default=...)
    started_timestamp: str = Field(description="The start timestamp", default=...)
    finished_timestamp: str = Field(description="The finish timestamp", default=...)
    clustering_algorithm: str = Field(description="The clustering algorithm used", default=...)
    clustering_params: dict[str, Any] = Field(description="The parameters for the clustering algorithm", default=...)
    preprocessing_params: PreProcessingParams | None = Field(
        description="The parameters for the preprocessing", default=...
    )
    labels: list[int | None] = Field(description="The labels for the dataset", default=...)
    additional_results: dict[str, Any] = Field(description="Additional results from the job", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c",
                    "dataset_name": "iris.csv",
                    "columns": [
                        {
                            "name": "sepal.length",
                            "type": "numeric",
                        },
                        {
                            "name": "sepal.width",
                            "type": "numeric",
                        },
                    ],
                    "created_timestamp": "2025-07-02T11:53:56.083632+00:00",
                    "started_timestamp": "2025-07-02T11:54:56.083632+00:00",
                    "finished_timestamp": "2025-07-02T11:55:56.083632+00:00",
                    "clustering_algorithm": "kmeans",
                    "clustering_params": {"n_clusters": 3},
                    "preprocessing_params": {"scaler": "standard"},
                    "labels": [0, 1, 2, 0, 1, 2],
                    "additional_results": {
                        "centers": [
                            [-0.11190209211560795, -0.9579796406026493],
                            [1.0961653346135656, 0.08900941628667573],
                            [-1.0020665312812713, 0.9062549154367601],
                        ],
                        "n_iter": 16,
                    },
                }
            ]
        }
    }


class ResultPostResponse(BaseModel):
    job_id: str = Field(description="The ID of the job", default=...)
    model_config = {"json_schema_extra": {"examples": [{"job_id": "fb231936-1d83-43de-85a4-81c6889dd21c"}]}}


@app.post("/result/")
async def post_result(req: ResultPostRequest, mongodb_database=Depends(get_mongodb)) -> ResultPostResponse:
    try:
        result_collection = mongodb_database.get_collection("results")
        # Debugging: Logge die zu speichernden Daten
        print(f"Saving result: {req.model_dump()}")
        await result_collection.insert_one(req.model_dump())
        return ResultPostResponse(job_id=req.job_id)
    except Exception as e:
        print(f"Error saving result: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving result: {e}") from e


class AutoMlClusterRequest(BaseModel):
    dataset_name: str = Field(..., description="The name of the dataset")
    columns: list[str] = Field(..., description="The columns to use for clustering")
    clustering_algorithms: list[str] | None = Field(
        default=None,
        description="List of clustering algorithms to use (e.g., KMeans, DBSCAN)",
    )
    dim_reduction_algorithms: list[str] | None = Field(
        default=None,
        description="List of dimensionality reduction algorithms (e.g., PCA, TSNE)",
    )
    evaluator_ls: list[str] | None = Field(
        default=None,
        description="List of clustering evaluation metrics (e.g., silhouetteScore)",
    )
    n_evaluations: int | None = Field(default=50, description="Number of AutoML evaluations to run")
    cutoff_time: int | None = Field(default=60, description="Time limit in seconds for each AutoML evaluation")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dataset_name": "iris.csv",
                    "columns": ["sepal.length", "sepal.width"],
                    "clustering_algorithms": ["KMeans", "DBSCAN"],
                    "dim_reduction_algorithms": ["PCA"],
                    "evaluator_ls": ["silhouetteScore", "calinskiHarabaszScore"],
                    "n_evaluations": 20,
                    "cutoff_time": 45,
                }
            ]
        }
    }


class AutoMlClusterResponse(BaseModel):
    job_id: str = Field(description="The ID of the AutoML clustering job", default=...)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "fb231936-1d83-43de-85a4-81c6889dd21c",
                }
            ]
        }
    }


@app.post("/automl/job")
async def start_automl(req: AutoMlClusterRequest) -> AutoMlClusterResponse:
    print("[AutoML] Received new request on /automl/job")

    try:
        task_kwargs = {
            "dataset_name": req.dataset_name,
            "columns": req.columns,
            "clustering_algorithms": req.clustering_algorithms,
            "dim_reduction_algorithms": req.dim_reduction_algorithms,
            "evaluator_ls": req.evaluator_ls,
            "n_evaluations": req.n_evaluations,
            "cutoff_time": req.cutoff_time,
        }

        job = celery.send_task("automl_worker.run_autocluster", kwargs=task_kwargs)

        print(f"[AutoML] Job started with ID: {job.id}")

        return AutoMlClusterResponse(job_id=job.id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting AutoML-Job: {str(e)}")


@app.get("/debug/job/{job_id}")
async def debug_job(job_id: str, mongodb_database=Depends(get_mongodb)):
    """
    Debug endpoint to check job status and results
    """
    try:
        # Check Celery task
        task = celery.AsyncResult(job_id)
        task_info = {
            "job_id": task.id,
            "status": task.status,
            "result": task.result if task.ready() else None,
        }

        # Check MongoDB
        results_collection = mongodb_database.get_collection("results")
        stored_result = await results_collection.find_one({"job_id": job_id})

        return {"task_info": task_info, "stored_result": stored_result is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}") from e


@app.get("/parameters/{algorithm_name}")
async def get_default_algorithm_parameters(
    algorithm_name: str = Path(
        description="The clustering algorithm to check for default parameters",
        default=...,
        examples=algorithms,
    ),
) -> dict[str, Any]:
    """
    Get the default parameters for a specific clustering algorithm.
    """
    algorithm = ALGORITHM_MAP.get(algorithm_name)
    if not algorithm:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    return {"clustering_params": algorithm.get_default_params()}


@app.get("/algorithms/")
async def get_available_algorithms() -> list[str]:
    """
    Get a list of available clustering algorithms.
    """
    return algorithms
