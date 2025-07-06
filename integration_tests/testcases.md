# Testcases - Integration Testing

## Frontend (Streamlit)

1. Upload new dataset -> It appears in the list
1. Upload dataset with already existing name -> Show error
1. Select a dataset and -> Its columns are shown
1. Delete a dataset and -> It disappears
1. Unselect all columns of chosen dataset -> Show error and hide "Clustering starten"
1. Select "Alle Spalten verwenden" -> All columns from dataset are shown
1. Select Clustering Algorithm -> Appropriate Parameters are shown
1. Click "Clustering starten" -> Job-ID is shown
1. Clustering successful -> Show as successful list overview with other recent clustering jobs
1. Clustering failed -> Show as failed in list overview
1. Click clustering job in list overview -> Show error message from job
1. Select clustering result from result list and select presentation form and click "Ergebnis anzeigen" -> Result is presented

## Backend (FastAPI)
