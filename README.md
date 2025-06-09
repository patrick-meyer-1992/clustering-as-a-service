# Clustering-as-a-Service (CaaS)
Eine RESTful-API, die es ermöglicht Datensätze zu clustern und die Ergebnisse zurück zu liefern.


## Pre-Commit Hooks
Um die Lesbarkeit des Codes zu verbessern und um einige Bugs frühzeitig zu vermeiden, sind für dieses Projekt pre-commit hooks implementiert.

Die pre-commit hooks prüfen und korrigieren teils automatisch die Formatierung des Codes.
Darüber hinaus werden erste Qualitätsprüfung des Codes vorgenommen. (sog. Linting).
Zuletzt findet auch eine Rechtschreibprüfung (nur Englisch) statt.

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
