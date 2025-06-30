# Clustering-as-a-Service (CaaS)
Eine RESTful-API, die es ermöglicht Datensätze zu clustern und die Ergebnisse zurück zu liefern.

Die Anwendung besteht aus mehreren Microservices, die auf Servern der Fernuniversität in Hagen (FUH) in einem Kubernetes Cluster betrieben werden.

## Voraussetzungen
Um diese Anwendung zu nutzen, sind einige Konfiguration vorzunehmen.

Da sie auf Servern der FUH gehostet wird, **ist ein Zugang zum Netzwerk der FUH ggf. über VPN notwendig**.

Innerhalb der FUH Netzes ist die Anwendung unter folgenden URLs erreichbar:

### Staging
http://app.staging.caas.local (Frontend)

http://api.staging.caas.local/docs (Backend-Dokumentation)

### Production
http://app.caas.local (Frontend)

http://api.caas.local/docs (Backend-Dokumentation)

Da die Anwendung und deren URLs nicht beim einem DNS-Server der FUH eingetragen sind, müssen lokale Änderungen zur Namensauflösung vorgenommen werden.

Die folgenden Zeilen müssen in die entsprechende Konfigurationsdatei eingetragen werden:
```
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

#### Unix

Eintragen der o.a. Zeilen in die Datei `/etc/hosts` (ggf. als `sudo`).


In jedem Fall ist eine Verbindung zum FUH Netz erforderlich, damit die Anwendung unter den aufgeführten URLS erreicht werden kann.


## Pre-Commit Hooks
Um die Lesbarkeit des Codes zu verbessern und um einige Bugs frühzeitig zu vermeiden, sind für dieses Projekt pre-commit hooks implementiert.

Die pre-commit hooks prüfen und korrigieren teils automatisch die Formatierung des Codes.
Darüber hinaus werden erste Qualitätsprüfung des Codes vorgenommen. (sog. Linting).

### Installation

#### Optional: Erstellen einer virtuellen Umgebung
Damit die benötigten Pakete nicht global sondern nur in diesem Projekt installiert werden, empfiehlt sich die Erstellung einer virtuellen Python-Umgebung (venv).
Es gibt verschiedene Möglichkeiten, diese zu erstellen.

**Kommandozeile**

Im Stammverzeichnis des Projekts `python -m venv .venv` ausführen.

Aktivierung der virtuellen Umgebung mit `source .venv/bin/activate`(Kann je nach Betriebssystem variieren).

Offizielle Dokumentation: https://docs.python.org/3/library/venv.html

**Visual Studio Code**

Öffnen der Suche mit `Strg+Shift+P`. Dort suchen nach `Python: Create Environment`.
Bei der Frage nach "environment type" `Venv` auswählen.

Die virtuelle Umgebung sollte automatisch durch VS Code aktiviert werden.

#### Installieren der benötigten Pakete
Im Stammverzeichnis des Projekts `pip install -r requirements.txt`ausführen. Dadurch werden die benötigten Pakete installiert.

### Workflow
Die hooks müssen einmalig mit `pre-commit install` in das Repository installiert werden

Die Tests können mit `pre-commit run --all-files` gestartet werden.

Zusätzlich werden sie automatisch bei jedem `git commit` ausgeführt.

### Was tun bei Fehlern?

**Fehler beheben**

Wenn es sich um eine sinnvolle Anmerkung handelt, sollte diese im Code eingepflegt werden. Danach kann der Commit erneut ausgeführt werden.

**Fehlertyp ausschließen**

In der Datei `pyproject.toml` können in das Array `lint.ignore` IDs von Fehlertypen ergänzt werden, die ignoriert werden sollen. IDs sollten nur dann ergänzt werden, wenn der Fehlertyp für dieses Projekt nicht angebracht ist und daher regelmäßig falsch-positive Ergebnisse erzeugt.

**Fehler ignorieren**

Wenn der Fehler aktuell nicht behoben werden kann, aber auch nicht als generelle Ausnahme ergänzt werden soll, können die Tests für diesen Commit mit `--no-verify` ausgesetzt werden, z.B. `git commit -m "add new feature" --no-verify`.


# Bzgl mongodb storage
minikube addons enable storage-provisioner
minikube addons enable default-storageclass

# Secrets apply
# Helm install mongodb
helm install mongodb bitnami/mongodb  -f values.yaml

# MongoDB erreichbar machen 
kubectl port-forward --address=0.0.0.0 svc/mongodb 32480:27017