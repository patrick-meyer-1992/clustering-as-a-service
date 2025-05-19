import streamlit as st
import pandas as pd
import requests
import plotly.express as px


url = "http://fastapi:8000/upload"

st.set_page_config(page_title="Clustering-as-a-Service", layout="wide")
st.title("Clustering-as-a-Service \U0001F50D")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vorschau deiner Daten:", df.head())

    if st.button("Clustering starten"):
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(url, files=files)

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
