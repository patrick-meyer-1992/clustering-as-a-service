# Clustering-as-a-Service (CaaS)
Eine RESTful-API, die es ermöglicht Datensätze zu clustern und die Ergebnisse zurück zu liefern.

Die Anwendung besteht aus mehreren skalierbaren Microservices, die auf Servern der Fernuniversität in Hagen (FUH) in einem Kubernetes Cluster betrieben werden.

## Clustering-Algorithmen
Das Clustering-Modul dieser Anwendung stellt eine Vielzahl an gebräuchlichen Clustering-Algorithmen zur Verfügung, die durch eine einheitliche Schnittstelle gekapselt sind. Jeder Algorithmus ist in einem eigenen Wrapper implementiert, der sowohl standardisierte Qualitätsmetriken berechnet als auch algorithmenspezifische Ausgaben (z. B. Clusterzentren, Wahrscheinlichkeiten, Anzahl der Iterationen) bereitstellt.

### Unterstützte Algorithmen

Folgende Algorithmen stehen aktuell zur Verfügung:

- **KMeans**  
- **Mini Batch KMeans**  
- **Bisecting KMeans**  
- **DBSCAN**  
- **HDBSCAN**  
- **OPTICS**  
- **Mean Shift**  
- **Agglomerative Clustering**  
- **Spectral Clustering**  
- **Affinity Propagation**  
- **BIRCH**  
- **Gaussian Mixture Model (GMM)**  
- **Bayesian Gaussian Mixture Model (BGMM)**

### Preprocessing

Vor der Anwendung eines Clustering-Verfahrens kann der Datensatz optional vorverarbeitet werden. Die Vorverarbeitung umfasst:

- **Imputation** (mittels Mittelwert oder Median)  
- **Outlier Removal** (Z-Score oder IQR-basiert)  
- **Feature Selection** (Konstant oder Niedrigvarianz)  
- **Skalierung** (Standard, MinMax, Robust, MaxAbs, oder automatisch)  
- **Normalisierung** (L1, L2, Max-Norm)  
- **Dimensionale Reduktion** via PCA  
- **Transformation** (Power- oder Quantile-Transformation)

Alle Schritte sind individuell konfigurierbar über ein separates Parameterobjekt und können bei Bedarf deaktiviert werden.

### Ergebnisstruktur

Die Ausgaben eines Clustering-Durchlaufs beinhalten:

- Cluster-Labels für alle Datenpunkte  
- Algorithmenspezifische Metadaten (z. B. Zentren, Wahrscheinlichkeiten)  
- Qualitätsmetriken:
  - Silhouette Score  
  - Davies-Bouldin Score  
  - Calinski-Harabasz Score  

Die Ergebnisse werden zusammen mit den genutzten Parametern und Zeitstempeln an den FastAPI-Backend gesendet und persistiert.


## AutoML
Die AutoML-Komponente ermöglicht es, automatisch geeignete Clustering-Konfigurationen für einen gegebenen Datensatz zu finden. Dazu werden verschiedene Clustering-Algorithmen, Parameterkombinationen und Evaluationsmetriken ausprobiert. Die vielversprechendsten Ergebnisse werden anschließend bereitgestellt.
Die Logik basiert auf dem [autocluster](https://github.com/wywongbd/autocluster) Projekt von Wong et al. Dieses Projekt verfolgt einen randomisierten Ansatz zur automatisierten Clustering-Konfiguration und kombiniert dabei mehrere Clustering-Verfahren, Dimensionalitätsreduktionen und Metriken. Es wurde um ein Frontend erweitert und einige Parameter wurden konfigurierbar gemacht.

# API-Endpunkt: AutoML starten

Ein AutoML-Job wird über folgenden Endpunkt gestartet:

```
POST /automl/cluster/
```

Dabei wird ein JSON-Request gesendet, das die zu analysierenden Spalten und weitere optionale Einstellungen enthält.
Beispiel-Request:

```json
{
  "dataset_name": "iris.csv",
  "columns": [
    {
      "name": "sepal.length",
      "type": "numeric"
    },
    {
      "name": "sepal.width",
      "type": "numeric"
    }
  ],
  "clustering_algorithms": ["KMeans", "DBSCAN"],
  "dim_reduction_algorithms": ["PCA"],
  "evaluator_ls": ["silhouetteScore", "calinskiHarabaszScore"],
  "n_evaluations": 20,
  "cutoff_time": 45,
  "clustering_num": [1, 10],
  "min_proportion": 0.01,
  "min_relative_proportion": "default"
}
```

Parameterbeschreibung:
| Parameter                  | Typ               | Beschreibung                                                                   |
| -------------------------- | ----------------- | ------------------------------------------------------------------------------ |
| `dataset_name`             | string            | Name eines zuvor hochgeladenen Datensatzes                                     |
| `columns`                  | List\[Object]     | Spaltenauswahl, inkl. Typ (`numeric` oder `categorical`)                       |
| `clustering_algorithms`    | List\[string]     | Liste der Clustering-Algorithmen (z. B. `KMeans`, `DBSCAN`)                    |
| `dim_reduction_algorithms` | List\[string]     | Liste der Verfahren zur Dimensionsreduktion (z. B. `PCA`, optional)            |
| `evaluator_ls`             | List\[string]     | Evaluationsmetriken, z. B. `silhouetteScore`, `calinskiHarabaszScore`          |
| `n_evaluations`            | int               | Anzahl zufälliger Konfigurationen, die getestet werden                         |
| `cutoff_time`              | int (Sekunden)    | Maximal erlaubte Laufzeit des AutoML-Prozesses                                 |
| `clustering_num`           | \[int, int]       | Intervall der Clusteranzahl (z. B. `[1, 10]`) für Algorithmen mit `n_clusters` |
| `min_proportion`           | float             | Minimale absolute Größe eines Clusters (z. B. 0.01 → mind. 1 %)                |
| `min_relative_proportion`  | float oder string | Minimale relative Größe eines Clusters (z. B. 0.05 oder `"default"`)           |


## Zugang zu CaaS bei FUH
Um diese Anwendung zu nutzen, sind einige Konfiguration vorzunehmen.

Da sie auf Servern der FUH gehostet wird, **ist ein Zugang zum Netzwerk der FUH ggf. über VPN notwendig**.

Innerhalb der FUH Netzes ist die Anwendung unter folgenden URLs nach u.a. Konfiguration erreichbar:

### Staging
http://app.staging.caas.local (Frontend)

http://api.staging.caas.local/docs (Backend-Dokumentation)

### Production
http://app.caas.local (Frontend)

http://api.caas.local/docs (Backend-Dokumentation)

### ArgoCD
https://argocd.local (Cluster-Management)

Da die Anwendung und deren URLs nicht bei einem DNS-Server der FUH eingetragen sind, müssen lokale Änderungen zur Namensauflösung vorgenommen werden.

Die folgenden Zeilen müssen in die entsprechende Konfigurationsdatei eingetragen werden:
```
132.176.108.158 argocd.local
132.176.108.158 app.staging.caas.local
132.176.108.158 api.staging.caas.local
132.176.108.158 app.caas.local
132.176.108.158 api.caas.local
```

#### Windows 11
Öffnen der Datei `C:\Windows\System32\drivers\etc\hosts` als Administrator mit einem Texteditor.

Dort o.a. Zeilen eintragen.

Ggf. ist ein Neustart erforderlich bevor die Änderungen wirksam werden.

#### Mac
Eingabe von `sudo nano /etc/hosts` in das Terminal.
In die geöffnete Datei die o.a. Zeilen eintragen.

#### Linux

Eintragen der o.a. Zeilen in die Datei `/etc/hosts` (ggf. als `sudo`).

In jedem Fall ist eine Verbindung zum FUH Netz erforderlich, damit die Anwendung unter den aufgeführten URLs erreicht werden kann.

Beim erstmaligen Aufrufen der URLs im Browser muss voraussichtlich eine Sicherheitsausnahmeregel akzeptiert werden, da die
Services kein gültiges Zertifikat anbieten.

## Lokaler Betrieb und Entwicklung
Zusätzlich zum Aufrufen auf den Servern der FUH, kann die Anwendung auch lokal betrieben werden.

Der lokale Betrieb ist auf Windows, Mac oder Linux möglich.
Es muss lediglich [**docker**](https://www.docker.com/get-started/ ) (inkl. **docker compose**) auf dem System installiert sein.

1. Klonen dieses Repositories
1. Umgebungsvariablen aktualisieren. An mehreren Stellen müssen dazu .env-Dateien erstellt werden. Der Inhalt für die .env-Dateien kann jeweils aus der danebenliegenden .env.example-Datei kopiert werden. Folgende Dateien sind zu erstellen: `app/api/.env` `message_broker/.env` `/mongodb/.env` `/redis/.env` 
1. Terminal öffnen und zum gerade geklonten Repository navigieren
1. `docker compose up -d --build`
1. Frontend aufrufen über http://localhost:8501
1. Backend-Dokumentation aufrufen über http://localhost:7001
1. Ggf. die Anwendung wieder stoppen mit `docker compose down`


Bei dieser Variante muss beachtet werden, dass die Microservices nicht skalierbar sind.
