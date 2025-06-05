import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objects as go
import io

FASTAPI_URL = "http://caas-fastapi:8000"

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

# --- Upload Section ---
uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

# Username für alle Aktionen
user_id = st.text_input("User-ID", value="testuser")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau deiner Daten:", df.head())
    
    if st.button("Datei speichern"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        data = {"user_id": user_id}  # User-ID wird mitgeschickt
        try:
            res = requests.put(f"{FASTAPI_URL}/upload/", files=files, data=data)
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
st.write("")  # Kleinerer Abstand als mit st.markdown("---")
dataset_list = get_dataset_list()
if dataset_list:
    # Container für einheitliches Styling
    with st.container():
        # Header-Zeile
        header_cols = st.columns([3, 2, 1, 1])
        header_cols[0].markdown("**Datensatzbezeichnung**")
        header_cols[1].markdown("**Benutzer**")
        
        # Trennlinie nach Header
        st.divider()
        
        # Für jeden Datensatz eine Zeile mit gleichmäßiger Aufteilung
        for dataset in dataset_list:
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            # Dateiname und Benutzer in gleicher Zeile
            col1.text(dataset['dataset_name'])
            col2.text(dataset.get('user_id', 'unbekannt'))
            
            # Buttons mit einheitlichem Styling, ohne Umbruch
            if col3.button("✅", key=f"select_{dataset['dataset_name']}", help="Auswählen"):
                st.session_state["dataset_name"] = dataset['dataset_name']
                st.success(f"Datensatz ausgewählt: {dataset['dataset_name']}")
            
            if col4.button("🗑️", key=f"delete_{dataset['dataset_name']}", help="Löschen"):
                if delete_dataset_backend(dataset['dataset_name']):
                    st.success(f"Datensatz gelöscht: {dataset['dataset_name']}")
                    st.rerun()
                else:
                    st.error("Fehler beim Löschen!")
            
            # Trennlinie zwischen den Datensätzen
            st.divider()
else:
    st.info("Keine Datensätze vorhanden")

# --- Clustering Section ---
if st.session_state["dataset_name"]:
    st.subheader("Clustering starten")
    
    # Versuche die Spalten aus dem Datensatz zu laden
    try:
        if uploaded_file and st.session_state["dataset_name"] == uploaded_file.name:
            available_columns = df.columns.tolist()
        else:
            # Lade den Datensatz aus dem Backend
            resp = requests.get(f"{FASTAPI_URL}/dataset/{st.session_state['dataset_name']}")
            if resp.status_code == 200:
                temp_df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')))
                available_columns = temp_df.columns.tolist()
            else:
                available_columns = []
        
        if available_columns:
            use_all_columns = st.checkbox("Alle Spalten verwenden", value=True)
            if use_all_columns:
                columns = available_columns
                st.info(f"Verwende alle Spalten: {', '.join(columns)}")
            else:
                columns = st.multiselect(
                    "Spalten für Clustering auswählen",
                    options=available_columns,
                    default=available_columns
                )
                if not columns:
                    st.warning("Bitte mindestens eine Spalte auswählen!")
        else:
            st.error("Keine Spalten im Datensatz gefunden!")
            columns = []
    
    except Exception as e:
        st.error(f"Fehler beim Laden der Spalten: {e}")
        columns = []

    clustering_algorithm = st.selectbox("Clustering-Algorithmus", ["kmeans", "dbscan"])
    preprocess = st.checkbox("Preprocessing", value=True)
    params = {}

    if st.button("Clustering starten") and columns:
        data = {
            "dataset_name": st.session_state["dataset_name"],
            "columns": json.dumps(columns),
            "clustering_algorithm": clustering_algorithm,
            "preprocess": str(preprocess),
            "user_id": user_id,  # User-ID wird auch beim Clustering mitgegeben
            "params": json.dumps(params)
        }
        try:
            res = requests.post(f"{FASTAPI_URL}/cluster/", data=data)
            if res.status_code == 200:
                response = res.json()
                st.session_state["job_id"] = response.get("job_id", "")
                st.success(f"Clustering gestartet! Job-ID: {st.session_state['job_id']}")
            else:
                st.error(f"Fehler beim Clustering: {res.text}")
        except Exception as e:
            st.error(f"Fehler beim Clustering: {e}")

# Presentation Section
# After asynchron clustering job is ended, the results get saved in mongodb. 
# The user can enter the Job-ID to retrieve the results.
st.subheader("Ergebnisse anzeigen")

# enter Job-ID (Upload or manuel)
input_job_id = st.text_input("Job-ID eingeben", value=st.session_state["job_id"])

presentation = st.selectbox(
    "Wie sollen Ergebnisse präsentiert werden?",
    ["Tabelle", "Rohdaten", "Graph"]
)

# get results over GET cluster function of FastAPI
if st.button("Ergebnis anzeigen") and input_job_id:
    
    mapping = {"Tabelle": "table", "Rohdaten": "raw", "Graph": "graph"}
    pres = mapping[presentation]
    
    # get data for presentation as requiered
    url = f"{FASTAPI_URL}/cluster/{input_job_id}?presentation={pres}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
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
                    st.plotly_chart(fig)
        else:
            try:
                msg = resp.json().get("detail", resp.text)
            except Exception:
                msg = resp.text
            st.error(f"Fehler: {msg}")
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Ergebnisse: {e}")