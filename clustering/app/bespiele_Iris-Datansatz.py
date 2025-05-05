# Bibliotheken importieren
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Schritt 1: Iris-Datensatz einlesen
df = pd.read_csv("Iris.csv")

# Schritt 2: Nur numerische Spalten verwenden (Id und Species ausschließen)
X = df.drop(columns=["Id", "Species"])

# Schritt 3: Daten skalieren (besonders wichtig für DBSCAN & Agglomerative)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Schritt 4: Drei verschiedene Clustering-Algorithmen definieren
algorithmen = {
    "KMeans": KMeans(n_clusters=3, random_state=0),
    "DBSCAN": DBSCAN(eps=0.8, min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=3)
}

# Schritt 5: Ergebnisse visualisieren
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, modell) in zip(axes, algorithmen.items()):
    labels = modell.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="tab10", s=60, edgecolors="black", linewidths=0.5)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.suptitle("Clustering des Iris-Datensatzes mit drei Algorithmen", fontsize=14, y=1.05)
plt.show()
