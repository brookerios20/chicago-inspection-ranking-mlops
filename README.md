# Chicago Restaurant Inspection Prioritization

This project is an MLOps pipeline that helps the City of Chicago decide which restaurants to inspect first. Using the city's public food inspection records from 2010 to the present, the model predicts the probability that an establishment will fail its next inspection, based on inspection history, facility type, risk category, neighborhood, time since the last visit, and nearby complaints. Chicago has a limited number of inspectors, so the system ranks establishments and is evaluated by how many failures it finds in the top k visits (precision@k). The city's 2018 food code overhaul changed its violation codes, which gives the monitoring system a real schema and concept drift event to detect.

*ADSP 32021 MLOps and Inference Engineering, University of Chicago M.S. Applied Data Science*

--- 

This README is a placeholder, and the setup instructions, architecture, and results will be added as the project develops.

## Data

The project uses the City of Chicago [Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) dataset from the Chicago Data Portal (dataset ID `4ijn-s7e5`). The analysis uses an export downloaded on 2026-11-02, which holds 317,961 inspections from 2010-01-04 to 2026-10-29. The CSV is about 340 MB, which is over GitHub's file size limit, so it is kept out of the repository.

To get the data:

1. Open the dataset page and choose Export, then CSV.
2. Save the file as `data/Food_Inspections_20261102.csv` in the repository root, which is the path the EDA notebook reads.

The portal updates the dataset as new inspections are recorded, so a later export will include more rows than the counts above.
