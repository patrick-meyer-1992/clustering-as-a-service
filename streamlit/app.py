import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

url = "http://caas-fastapi:8000/upload"

st.set_page_config(page_title="Clustering-as-a-Service", layout="wide")
st.title("Clustering-as-a-Service \U0001F50D")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])


# Upload Section
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau deiner Daten:", df.head())

    if st.button("Clustering starten"):
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(url, files=files)

        # If upload successful expand the resultpage
        if res.status_code == 200:
            clusters = res.json()["clusters"]
            df["Cluster"] = clusters
            st.success("Clustering abgeschlossen!")
            st.write(df)

            # Visualisierung
            if len(df.columns) >= 3:
                st.subheader("2D-Visualisierung")
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df["Cluster"].astype(str))
                st.plotly_chart(fig)
        else:
            st.error("Fehler beim Clustering")

# Results by task id

task_id = st.text_input("Job-ID eingeben")
presentation = st.selectbox(
    "Wie sollen Ergebnisse präsentiert werden?",
    ["Tabelle", "Rohdaten", "Graph"]
)

# Results Section
st.subheader("Ergebnisse anzeigen")
if st.button("Ergebnis anzeigen") and task_id:
    mapping = {"Tabelle": "table", "Rohdaten": "raw", "Graph": "graph"}
    pres = mapping[presentation]
    url = f"http://caas-fastapi:8000/cluster/{task_id}?presentation={pres}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        
        # Check if the response is table, raw, or graph
        if pres == "table":
            st.write(pd.DataFrame({"labels": data["labels"]}))
        elif pres == "raw":
            st.write(np.array(data))
        elif pres == "graph":
            fig = go.Figure(data)
            st.plotly_chart(fig)
    else:
        st.error(f"Fehler: {resp.text}")