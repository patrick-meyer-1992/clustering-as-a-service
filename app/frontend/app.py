import os
import sys
from typing import Literal, get_args

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clustering import wrappers
from clustering.preprocessing_params import PreProcessingParams
from utils.config import FASTAPI_HOST, FASTAPI_PORT, FASTAPI_PROTOCOL
from utils.logger import setup_logger

logger = setup_logger(__name__)

FASTAPI_URL = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}"

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = ""
if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""


def get_dataset_list():
    """
    Retrieve the list of available datasets from the backend API.

    Returns:
        list: A list of dataset information dictionaries.
    """
    try:
        resp = requests.get(f"{FASTAPI_URL}/datasets/")
        if resp.status_code == 200:
            return resp.json()
        else:
            return []
    except Exception:
        return []


def delete_dataset_backend(dataset_name):
    """
    Delete a dataset in the backend by its name.

    Args:
        dataset_name (str): The name of the dataset to delete.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    try:
        resp = requests.delete(f"{FASTAPI_URL}/dataset/{dataset_name}")
        return resp.status_code == 200
    except Exception:
        return False


def get_available_clustering_algorithms():
    """
    Dynamically loads all clustering algorithms from the clustering directory.
    Excludes abstract base classes.

    Returns:
        dict: Dictionary mapping frontend algorithm names to their wrapper classes.
    """
    algorithms = {
        getattr(wrappers, algo).frontend_name: getattr(wrappers, algo)
        for algo in dir(wrappers)
        if algo.endswith("Wrapper")
    }
    return algorithms


def get_backend_frontend_mapping():
    """
    Dynamically loads the frontend name for each backend name from the clustering directory.
    Excludes abstract base class.
    """

    mapping = {
        getattr(wrappers, algo).backend_name: getattr(wrappers, algo).frontend_name
        for algo in dir(wrappers)
        if algo.endswith("Wrapper")
    }

    mapping["AutoML"] = "AutoML"  # Add AutoML mapping

    return mapping


def parse_params_value(value) -> str | int | float | bool | None:
    """
    Parse a string parameter value into its appropriate type.

    Args:
        value (str): The value to parse.

    Returns:
        str | int | float | bool | None: The parsed value.
    """
    if not isinstance(value, str):
        return value
    if value.lower() == "none":
        return None
    elif value.lower() in ["true", "false"]:
        return value.lower() == "true"
    elif value.isdigit():
        return int(value)
    elif value.replace(".", "", 1).isdigit():
        return float(value)
    else:
        return value


# --- Upload Section ---
uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau deiner Daten:", df.head())

    if st.button("Datei speichern"):
        # Prepare file for upload
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        try:
            res = requests.put(f"{FASTAPI_URL}/dataset/", files=files)
            if res.status_code == 200:
                response = res.json()
                st.session_state["dataset_name"] = response.get("dataset_name", "")
                st.success(f"Datei gespeichert: {st.session_state['dataset_name']}")
                st.rerun()
            elif res.status_code == 409:
                st.error("Ein Datensatz mit diesem Namen existiert bereits!")
            else:
                st.error(f"Fehler beim Speichern: {res.text}")
        except Exception as e:
            st.error(f"Fehler beim Speichern: {e}")

# --- Dataset List Section ---
st.subheader("Vorhandene Datensätze")
dataset_list = get_dataset_list()
if dataset_list:
    # Create container for consistent styling
    with st.container():
        # Header row
        header_cols = st.columns([3, 2, 1, 1])
        header_cols[0].markdown("**Datensatzbezeichnung**")

        st.divider()

        # Create a row for each dataset with uniform spacing
        for dataset in dataset_list:
            col1, col3, col4 = st.columns([5, 1, 1])

            # Display filename
            col1.text(dataset["dataset_name"])

            # Action buttons
            if col3.button("✅", key=f"select_{dataset['dataset_name']}", help="Auswählen"):
                st.session_state["dataset_name"] = dataset["dataset_name"]
                st.success(f"Datensatz ausgewählt: {dataset['dataset_name']}")

            if col4.button("🗑️", key=f"delete_{dataset['dataset_name']}", help="Löschen"):
                if delete_dataset_backend(dataset["dataset_name"]):
                    st.success(f"Datensatz gelöscht: {dataset['dataset_name']}")
                    st.rerun()
                else:
                    st.error("Fehler beim Löschen!")

            st.divider()
else:
    st.info("Keine Datensätze vorhanden")

# --- Clustering Section ---
if st.session_state["dataset_name"]:
    st.subheader("Clustering starten")

    # Load columns and types from backend
    available_columns = []
    try:
        resp = requests.get(f"{FASTAPI_URL}/metadata/{st.session_state['dataset_name']}")
        available_columns = resp.json().get("columns") if resp.status_code == 200 else []

    except Exception as e:
        st.error(f"Fehler beim Laden der Spalteninformationen: {e}")
        available_columns = []

    if available_columns:
        st.markdown("**Spaltenauswahl**")

        if "column_selection" not in st.session_state or st.session_state["dataset_name"] != st.session_state.get(
            "last_column_selection_dataset", ""
        ):
            st.session_state["column_selection"] = {
                column["name"]: {
                    "use": True,
                    "selected_type": column["allowed_types"][0],
                    "allowed_types": column["allowed_types"],
                }
                for column in available_columns
            }
            st.session_state["last_column_selection_dataset"] = st.session_state["dataset_name"]

        all_selected = all(
            st.session_state["column_selection"][col]["use"] for col in st.session_state["column_selection"]
        )
        select_all = st.checkbox("Alle Spalten auswählen", value=all_selected, key="select_all_columns")

        # Synchronize individual checkboxes if "Select All" was changed
        if select_all != all_selected:
            for col in st.session_state["column_selection"]:
                st.session_state["column_selection"][col]["use"] = select_all

        # Table header
        header = st.columns([1, 3, 2])
        header[0].markdown("**Verwenden**")
        header[1].markdown("**Spaltenname**")
        header[2].markdown("**Typ**")

        # Each column as its own row, always neatly aligned
        for col in available_columns:
            col_name = col["name"]
            use = st.session_state["column_selection"][col_name]["use"]
            row = st.columns([1, 3, 2])
            # Checkbox for use
            new_use = row[0].checkbox("Checkbox", value=use, key=f"use_{col_name}", label_visibility="hidden")
            st.session_state["column_selection"][col_name]["use"] = new_use
            # Column name
            row[1].markdown(f"{col_name}")
            new_type = row[2].selectbox(
                "Selectbox",
                options=st.session_state["column_selection"][col_name]["allowed_types"],
                index=st.session_state["column_selection"][col_name]["allowed_types"].index(
                    st.session_state["column_selection"][col_name]["selected_type"]
                ),
                key=f"type_{col_name}",
                label_visibility="hidden",
            )
            st.session_state["column_selection"][col_name]["selected_type"] = new_type

    else:
        st.error("Keine Spalten im Datensatz gefunden!")

    # Columns for Clustering
    columns = [
        {"name": col, "type": st.session_state["column_selection"][col]["selected_type"]}
        for col in st.session_state["column_selection"]
        if st.session_state["column_selection"][col]["use"]
    ]

    use_automl = st.checkbox("AutoML verwenden (automatische Algorithmusauswahl)", value=False)

    selected_cluster_algorithms = []
    selected_dim_reduction = []
    selected_evaluators = []

    if use_automl:
        # Multiple selection for clustering algorithms
        available_cluster_algorithms = [
            "KMeans",
            "GaussianMixture",
            "Birch",
            "MiniBatchKMeans",
            "AgglomerativeClustering",
            "SpectralClustering",
        ]
        selected_cluster_algorithms = st.multiselect(
            "Clustering-Algorithmen auswählen",
            available_cluster_algorithms,
            default=available_cluster_algorithms,
        )

        # Multiple selection for dimensionality reduction algorithms
        available_dim_reduction = [
            "TSNE",
            "PCA",
            "IncrementalPCA",
            "KernelPCA",
            "FastICA",
            "TruncatedSVD",
        ]
        selected_dim_reduction = st.multiselect(
            "Dimensionality Reduction auswählen",
            available_dim_reduction,
            default=available_dim_reduction,
        )

        # Multiple selection for evaluators
        available_evaluators = [
            "silhouetteScore",
            "daviesBouldinScore",
            "calinskiHarabaszScore",
        ]
        selected_evaluators = st.multiselect("Evaluator auswählen", available_evaluators, default=available_evaluators)

        n_evaluations = st.slider("Anzahl AutoML Evaluationen", min_value=10, max_value=200, value=50, step=10)
        cutoff_time = st.slider(
            "Maximale Laufzeit pro Evaluation (Sekunden)",
            min_value=10,
            max_value=300,
            value=60,
            step=10,
        )

    else:
        # Dynamic algorithm selection
        clustering_algorithms = get_available_clustering_algorithms()
        clustering_algorithm = st.selectbox(
            "Clustering-Algorithmus auswählen",
            options=sorted(clustering_algorithms.keys()),
        )
        clustering_params = clustering_algorithms.get(clustering_algorithm).get_default_params()
        chosen_clustering_params = {}
        with st.expander("Clustering-Parameter konfigurieren"):
            for k, v in clustering_params.items():
                chosen_clustering_params[k] = st.text_input(label=k, value=str(v))

        preprocess = st.checkbox("Preprocessing", value=True)
        preprocessing_params = PreProcessingParams().model_dump()
        chosen_preprocessing_params = {}
        allowed_values = {}
        for name, field in PreProcessingParams.model_fields.items():
            annotation = field.annotation
            # Only process fields with Literal type
            if getattr(annotation, "__origin__", None) is Literal:
                allowed_values[name] = get_args(annotation)

        if preprocess:
            with st.expander("Preprocessing-Parameter konfigurieren"):
                for k, v in preprocessing_params.items():
                    if k in allowed_values:
                        chosen_preprocessing_params[k] = parse_params_value(
                            st.selectbox(label=k, options=allowed_values[k])
                        )
                    elif isinstance(v, bool):
                        chosen_preprocessing_params[k] = parse_params_value(
                            st.selectbox(label=k, options=["False", "True"])
                        )
                    else:
                        chosen_preprocessing_params[k] = parse_params_value(st.text_input(label=k, value=str(v)))
                # chosen_preprocessing_params = PreProcessingParams(
                #     **chosen_preprocessing_params
                # )

    if st.button("Clustering starten"):
        dataset_name = st.session_state["dataset_name"]

        cluster_url = f"{FASTAPI_URL}/automl/job" if use_automl else f"{FASTAPI_URL}/job/"

        if use_automl:
            dim_algos = [algo for algo in selected_dim_reduction if algo != "Keine"]
            data = {
                "dataset_name": dataset_name,
                "columns": columns,
                "clustering_algorithms": selected_cluster_algorithms,
                "dim_reduction_algorithms": selected_dim_reduction or None,
                "evaluator_ls": selected_evaluators,
                "n_evaluations": n_evaluations,
                "cutoff_time": cutoff_time,
            }

        else:
            if preprocess:
                filtered_preprocessing_params = {k: v for k, v in chosen_preprocessing_params.items() if v is not None}
            else:
                filtered_preprocessing_params = None
            # === Manual path ===
            data = {
                "dataset_name": dataset_name,
                "columns": columns,
                "clustering_algorithm": clustering_algorithms.get(clustering_algorithm).backend_name,
                "preprocess": preprocess,
                "clustering_params": chosen_clustering_params,
                "preprocessing_params": (filtered_preprocessing_params),
            }

        try:
            res = requests.post(cluster_url, json=data)
            if res.status_code == 200:
                response = res.json()
                job_id = response.get("job_id", "")
                st.session_state["job_id"] = job_id
                st.success(f"Clustering gestartet! Job-ID: {job_id}")

            else:
                st.error("Fehler beim Starten des Clusterings")
        except Exception as e:
            st.error(f"Fehler: {str(e)}")


# Presentation Section
# After the asynchronous clustering job has ended, the results are saved in MongoDB.
# The user can enter the Job-ID to retrieve the results.
st.subheader("Ergebnisse anzeigen")

#
# Selection mode for job selection
job_select_mode = st.radio(
    label="Job auswählen", options=["Manuelle Eingabe", "Aktuellen Job anzeigen", "Job-Historie"], index=0
)

input_job_id = ""
if job_select_mode == "Manuelle Eingabe":
    input_job_id = st.text_input("Job-ID eingeben", value=st.session_state.get("job_id", ""))
elif job_select_mode == "Aktuellen Job anzeigen":
    # Zeige den aktuellsten Job aus dem Session-State
    input_job_id = st.session_state.get("job_id", "")
    if input_job_id:
        st.info(f"Aktueller Job: {input_job_id}")
    else:
        st.warning("Kein aktueller Job vorhanden.")
elif job_select_mode == "Job-Historie":

    def get_job_list():
        """
        Retrieve the list of clustering jobs from the backend API, including their live status.

        Returns:
            list: List of job information dictionaries, each updated with their current status.
        """

        resp = requests.get(f"{FASTAPI_URL}/jobs/")
        if resp.status_code == 200:
            jobs = resp.json()
            return jobs

    job_list = get_job_list()
    job_options = []
    job_id_to_label = {}
    if job_list:
        # Sort the list so that the most recent jobs are first
        job_list = sorted(job_list, key=lambda job: job.get("created_timestamp"), reverse=True)
        for job in job_list:
            algorithm_name = get_backend_frontend_mapping().get(job.get("clustering_algorithm"))
            label = f"{job['job_id']} | {job['dataset_name']} | {algorithm_name} | {job['status']}"
            job_options.append(label)
            job_id_to_label[label] = job["job_id"]
        selected_label = st.selectbox(
            "Vorherigen Job auswählen:",
            options=job_options,
        )
        input_job_id = job_id_to_label[selected_label]
    else:
        st.info("Keine Jobs vorhanden.")

presentation = ""
if job_select_mode != "Job-Historie" or selected_label.endswith(" | PERSISTED"):
    presentation = st.selectbox("Wie sollen Ergebnisse präsentiert werden?", ["Graph", "Tabelle"])

    if presentation == "Graph" and input_job_id:
        url = f"{FASTAPI_URL}/result/{input_job_id}/raw"
        response = requests.get(url, params={"field": "columns"})

        if response.status_code == 404:
            st.warning(
                ""
                "Entweder gibt es keine Jobs zu dieser ID "
                "oder die Ergebnisse sind noch nicht verfügbar. "
                "Bitte prüfen Sie, ob die ID gültig ist oder warten Sie, "
                "bis der Job abgeschlossen ist."
            )
            st.stop()
        else:
            available_result_columns = response.json()
            available_result_columns = [col.get("name") for col in available_result_columns]
            selected_result_columns = st.multiselect(
                "Welche Spalten sollen für den Graphen verwendet werden?",
                options=available_result_columns,
                default=available_result_columns[:2],
                max_selections=2,
            )

    if st.button("Ergebnis anzeigen") and input_job_id:
        mapping = {"Tabelle": "table", "Graph": "graph"}
        pres = mapping[presentation]
        url = f"{FASTAPI_URL}/result/{input_job_id}/{pres}"

        if pres == "graph" and len(selected_result_columns) != 2:
            st.error("Für den Graphen müssen genau zwei Spalten ausgewählt werden.")
            st.stop()
        try:
            params = (
                None
                if pres == "table"
                else {
                    "x_column": selected_result_columns[0],
                    "y_column": selected_result_columns[1],
                }
            )
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                # Präsentation auf voller Breite
                if pres == "table":
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                elif pres == "graph":
                    fig = go.Figure(data)
                    if not fig.data:
                        st.warning("Keine Daten für Plot vorhanden.")
                    else:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                try:
                    msg = resp.json().get("detail", resp.text)
                except Exception:
                    msg = resp.text
                st.error(f"Fehler: {msg}")
        except Exception as e:
            st.error(f"Fehler beim Abrufen der Ergebnisse: {e}")
else:
    st.warning("Die Ergebnisse sind noch nicht verfügbar. Bitte warten Sie, bis der Job abgeschlossen ist.")
    st.stop()
