import streamlit as st
import pandas as pd
import requests
import json

FASTAPI_URL = "http://caas-fastapi:8000"

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = ""
if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau deiner Daten:", df.head())
    user_id = st.text_input("User-ID", value="testuser")
    if st.button("Datei speichern"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        data = {"user_id": user_id}
        try:
            res = requests.put(f"{FASTAPI_URL}/upload/", files=files, data=data)
            if res.status_code == 200:
                response = res.json()
                st.session_state["dataset_name"] = response.get("dataset_name", "")
                st.success(f"Datei gespeichert: {st.session_state['dataset_name']}")
            else:
                st.error(f"Fehler beim Speichern: {res.text}")
        except Exception as e:
            st.error(f"Fehler beim Speichern: {e}")

if st.session_state["dataset_name"]:
    st.subheader("Clustering starten")
    columns = st.multiselect("Spalten für Clustering auswählen", df.columns.tolist())
    clustering_algorithm = st.selectbox("Clustering-Algorithmus", ["kmeans", "dbscan"])
    preprocess = st.checkbox("Preprocessing", value=True)
    params = {}  # ggf. weitere Parameter
    if st.button("Clustering starten"):
        data = {
            "dataset_name": st.session_state["dataset_name"],
            "columns": json.dumps(columns),
            "clustering_algorithm": clustering_algorithm,
            "preprocess": str(preprocess),
            "user_id": user_id,
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