import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from clustering import wrappers

FASTAPI_HOST = os.getenv("FASTAPI_HOST")
FASTAPI_PORT = os.getenv("FASTAPI_PORT")
FASTAPI_PROTOCOL = os.getenv("FASTAPI_PROTOCOL")
FASTAPI_URL = f"{FASTAPI_PROTOCOL}://{FASTAPI_HOST}:{FASTAPI_PORT}"

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = ""
if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""


def get_dataset_list():
    try:
        resp = requests.get(f"{FASTAPI_URL}/datasets/")
        if resp.status_code == 200:
            return resp.json()
        else:
            return []
    except Exception:
        return []


def delete_dataset_backend(dataset_name):
    try:
        resp = requests.delete(f"{FASTAPI_URL}/datasets/{dataset_name}")
        return resp.status_code == 200
    except Exception:
        return False


def get_available_clustering_algorithms():
    """
    Dynamically loads all clustering algorithms from the clustering directory.
    Excludes abstract base class.
    """
    algorithms = {
        getattr(wrappers, algo).frontend_name: getattr(wrappers, algo).backend_name
        for algo in dir(wrappers)
        if algo.endswith("Wrapper")
    }

    return algorithms


# --- Upload Section ---
uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau deiner Daten:", df.head())

    if st.button("Datei speichern"):
        # Prepare file for upload
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        try:
            res = requests.put(f"{FASTAPI_URL}/upload/", files=files)
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

    # Load columns from selected dataset
    try:
        if uploaded_file and st.session_state["dataset_name"] == uploaded_file.name:
            available_columns = df.columns.tolist()
        else:
            # Load dataset from backend if not currently uploaded
            resp = requests.get(f"{FASTAPI_URL}/dataset/{st.session_state['dataset_name']}")
            if resp.status_code == 200:
                temp_df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
                available_columns = temp_df.columns.tolist()
            else:
                available_columns = []

        # Column selection interface
        if available_columns:
            use_all_columns = st.checkbox("Alle Spalten verwenden", value=True)
            if use_all_columns:
                columns = available_columns
                st.info(f"Verwende alle Spalten: {', '.join(columns)}")
            else:
                columns = st.multiselect(
                    "Spalten für Clustering auswählen",
                    options=available_columns,
                    default=available_columns,
                )
                if not columns:
                    st.warning("Bitte mindestens eine Spalte auswählen!")
        else:
            st.error("Keine Spalten im Datensatz gefunden!")
            columns = []

    except Exception as e:
        st.error(f"Fehler beim Laden der Spalten: {e}")
        columns = []

    # Dynamic algorithm selection
    clustering_algorithms = get_available_clustering_algorithms()
    clustering_algorithm = st.selectbox(
        "Clustering-Algorithmus auswählen", options=sorted(clustering_algorithms.keys())
    )
    preprocess = st.checkbox("Preprocessing", value=True)
    clustering_params = {}
    preprocessing_params = {}

    if st.button("Clustering starten"):
        data = {
            "dataset_name": st.session_state["dataset_name"],
            "columns": columns,
            "clustering_algorithm": clustering_algorithms.get(clustering_algorithm),
            "preprocess": preprocess,
            "clustering_params": clustering_params,
            "preprocessing_params": preprocessing_params
        }

        print(data)

        try:
            res = requests.post(f"{FASTAPI_URL}/cluster/", json=data)
            if res.status_code == 200:
                response = res.json()
                job_id = response.get("job_id", "")
                st.session_state["job_id"] = job_id
                st.success(f"Clustering gestartet! Job-ID: {job_id}")

                # Fortschrittsanzeige
                status_placeholder = st.empty()
                progress = 0
                max_wait = 120  # max 2 Minuten warten
                poll_interval = 2  # alle 2 Sekunden abfragen

                for _ in range(max_wait // poll_interval):
                    debug_resp = requests.get(f"{FASTAPI_URL}/debug/job/{job_id}")
                    if debug_resp.status_code == 200:
                        debug_data = debug_resp.json()
                        celery_status = debug_data["task_info"]["status"]
                        if celery_status == "PENDING":
                            status_placeholder.info("Job ist in der Warteschlange (PENDING)...")
                        elif celery_status == "STARTED":
                            status_placeholder.info("Clustering läuft (STARTED)...")
                        elif celery_status == "SUCCESS":
                            status_placeholder.success("Clustering abgeschlossen!")
                            break
                        elif celery_status == "FAILURE":
                            status_placeholder.error("Clustering fehlgeschlagen!")
                            break
                        else:
                            status_placeholder.warning(f"Status: {celery_status}")
                    else:
                        status_placeholder.warning("Status konnte nicht abgefragt werden.")
                    time.sleep(poll_interval)
                else:
                    status_placeholder.warning("Timeout: Clustering hat zu lange gedauert.")
            else:
                st.error("Fehler beim Starten des Clusterings")
        except Exception as e:
            st.error(f"Fehler: {str(e)}")

# Presentation Section
# After asynchron clustering job is ended, the results get saved in mongodb.
# The user can enter the Job-ID to retrieve the results.
st.subheader("Ergebnisse anzeigen")

# Auswahlmodus für die Job-Auswahl
job_select_mode = st.radio("Job auswählen", ["Manuelle Eingabe", "Aktuellen Job anzeigen", "Job-Historie"])

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
        try:
            resp = requests.get(f"{FASTAPI_URL}/jobs/")
            if resp.status_code == 200:
                jobs = resp.json()
                # Status für jeden Job live abfragen
                for job in jobs:
                    job_id = job.get("job_id")
                    if job_id:
                        debug_resp = requests.get(f"{FASTAPI_URL}/debug/job/{job_id}")
                        if debug_resp.status_code == 200:
                            celery_status = debug_resp.json()["task_info"]["status"]
                            job["status"] = celery_status
                return jobs
            else:
                return []
        except Exception:
            return []

    job_list = get_job_list()
    job_options = []
    job_id_to_label = {}
    if job_list:
        # Sortiere die Liste so, dass die neuesten Jobs zuerst stehen
        job_list = sorted(job_list, key=lambda job: job.get("created_timestamp", ""), reverse=True)
        for job in job_list:
            label = f"{job['job_id']} | {job['dataset_name']} | {job['clustering_algorithm']} | {job['status']}"
            job_options.append(label)
            job_id_to_label[label] = job["job_id"]
        selected_label = st.selectbox(
            "Vorherigen Job auswählen:",
            options=job_options,
        )
        input_job_id = job_id_to_label[selected_label]
    else:
        st.info("Keine Jobs vorhanden.")

presentation = st.selectbox("Wie sollen Ergebnisse präsentiert werden?", ["Tabelle", "Rohdaten", "Graph"])

if st.button("Ergebnis anzeigen") and input_job_id:
    mapping = {"Tabelle": "table", "Rohdaten": "raw", "Graph": "graph"}
    pres = mapping[presentation]
    url = f"{FASTAPI_URL}/cluster/{input_job_id}/{pres}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            # Präsentation auf voller Breite
            if pres == "table":
                if "data" in data and "columns" in data:
                    df = pd.DataFrame(data["data"], columns=data["columns"])
                    st.write(df)
                elif "labels" in data:
                    st.write(pd.DataFrame({"labels": data["labels"]}))
                else:
                    st.warning("Keine Tabellendaten vorhanden.")
            elif pres == "raw":
                st.write(np.array(data))
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
