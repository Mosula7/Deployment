# Deployment

To run the container you need to follow the following steps:
* make an .env file in the Deployment directory with AIRFLOW_UID and AIRFLOW_GID, for example:
```
AIRFLOW_UID=50000
AIRFLOW_GID=0
```
**IMPORTANT:** If you are running the container on a linux machine make sure the files that are volumes have read permissions and for the models direcotry: write permission also, so it can be accessed from the container.\
**IMPORTANT:** If you want to run the container you either need to run it under GID 0 or UID 50000 (or both) \
Also, you will need to specify database config parameters in the env file. \
If you want to connect to the database that is being created in this projects docker compose file you will need to include the following configuration: 
```
DB_HOST="postgres"
DB_NAME="postgres"
DB_PORT="5432"
DB_USER="airflow"
DB_PASS="airflow"
```
This configuration will need to be changed if you want to connect to a different database  \
**IMPORTANT:** If you want to run the project directry on your machine without docker also copy .env file into the airflow/dags directory. \
When the project is run in docker a volume is created on .env file in the dags directory. 

* first, to initialize the postgres database run:
```
docker compose up airflow-init
```

* next run to run both airflow and the webserver with the form
```
docker compose up -d
```
* go to localhost 5000 for churn prediction form and 8080 for airflow (user and password is set to airflow, you can change this from the docker compose file)

* in the airflow before you start making any predictions from the form or in batch in airflow run config_db dag only once, which creates a database and all the neccecery tables for logging. This creates three tables: churn_data - historical data for client features and their churn status, churn_models - model performance logs when models are trained, churn_predictions: data that the model predicts on both batch and web form. In this project I'm using the same data for batch prediction as I'm training on, but in a real application this would be a stream of new data with unknown labels, batch predictions are logged with flag 'batch'. predictions that are made from the web form are also logged into the same table, but with flag web.
* IMPORTANT: RUN config_db before you run the churn_pipline or web prediction, as this dag creates all the neccesery tables for logging and without this it won't work.

* The second dag in airflow is responsible for processing raw data and uploading in to the database, model training and batch prediction.

* model_config.json - hyperparameters to train the model with.

* predict_config.json - name of the model used for batch and web form prediction, the model MUST be in the models folder.

* predict_churn.ipynb - notebook I used for testing

