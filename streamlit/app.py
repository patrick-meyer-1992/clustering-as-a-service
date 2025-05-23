import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
import json

FASTAPI_URL = "http://caas-fastapi:8000"

st.set_page_config(page_title="Clustering-as-a-Service", layout="wide")
st.title("Clustering-as-a-Service \U0001F50D")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

# init session state for Job-ID
if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""

# Upload Section
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau deiner Daten:", df.head())

    # read column names
    columns = list(df.columns)
    clustering_algorithm = st.selectbox("Clustering-Algorithmus", ["kmeans", "dbscan"])
    preprocess = st.checkbox("Preprocessing", value=True)
    user_id = st.text_input("User-ID", value="testuser")
    params = {}

    if st.button("Clustering starten"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        data = {
            "columns": json.dumps(columns),
            "clustering_algorithm": clustering_algorithm,
            "preprocess": str(preprocess),
            "user_id": user_id,
            "params": json.dumps(params)
        }
        try:
            res = requests.put(f"{FASTAPI_URL}/dataset/", files=files, data=data)
            if res.status_code == 200:
                response = res.json()
                st.session_state["job_id"] = response.get("job_id", "")
                st.success(f"Upload & Clustering gestartet! Job-ID: {st.session_state['job_id']}")
            else:
                st.error(f"Fehler beim Upload/Jobstart: {res.text}")
        except Exception as e:
            st.error(f"Fehler beim Upload/Jobstart: {e}")


# Presentation Section
st.subheader("Ergebnisse anzeigen")

# Enter Job-ID (Upload or manuel)
input_job_id = st.text_input("Job-ID eingeben", value=st.session_state["job_id"])

presentation = st.selectbox(
    "Wie sollen Ergebnisse präsentiert werden?",
    ["Tabelle", "Rohdaten", "Graph"]
)

if st.button("Ergebnis anzeigen") and input_job_id:
    
    mapping = {"Tabelle": "table", "Rohdaten": "raw", "Graph": "graph"}
    pres = mapping[presentation]
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
                import plotly.graph_objects as go
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