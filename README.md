# Chicago Restaurant Inspection Prioritization

This project is an MLOps pipeline that helps the City of Chicago decide which restaurants to inspect first. Using the city's public food inspection records from 2010 to the present, the model predicts the probability that an establishment will fail its next inspection, based on inspection history, facility type, risk category, neighborhood, time since the last visit, and nearby complaints. Chicago has a limited number of inspectors, so the system ranks establishments and is evaluated by how many failures it finds in the top k visits (precision@k). The city's 2018 food code overhaul changed its violation codes, which gives the monitoring system a real schema and concept drift event to detect.

This README is a placeholder, and the setup instructions, architecture, and results will be added as the project develops.
