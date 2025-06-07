from .celery_conn import celery
import pandas as pd
# from autocluster import run_pipeline

@celery.task(name="automl_worker.run_autocluster")
def run_autocluster(data_dict):
    df = pd.DataFrame(data_dict)
    #results = run_pipeline(df)
    #return results
    return "Hello AutoML Worker!"