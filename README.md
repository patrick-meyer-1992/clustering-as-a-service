# Clustering-as-a-Service (CaaS)
Eine RESTful-API, die es ermöglicht Datensätze zu clustern und die Ergebnisse zurück zu liefern.

Die Anwendung besteht aus mehreren skalierbaren Microservices, die auf Servern der Fernuniversität in Hagen (FUH) in einem Kubernetes Cluster betrieben werden.

## Clustering-Algorithmen

## AutoML

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