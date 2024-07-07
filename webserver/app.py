import catboost as cat
import os
from flask import Flask, render_template
import json
from web_utils import predict, result


with open('predict_config.json') as file:
    model_name = json.load(file)['model_name']

app = Flask(__name__)
model = cat.CatBoostClassifier()
model.load_model(os.path.join('models', model_name))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_port = os.getenv("DB_PORT")
    db_name = 'churn'
    return predict(model=model, 
                   model_name=model_name, 
                   db_host=db_host, 
                   db_name=db_name, 
                   db_user=db_user, db_pass=db_pass,
                   db_port=db_port)


@app.route('/result')
def result_route():
    return result()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
