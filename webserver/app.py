import catboost as cat
import os
from flask import Flask, render_template, request, redirect, url_for
import psycopg2
import json


with open('predict_config.json') as file:
    model_name = json.load(file)['model_name']

app = Flask(__name__)
model = cat.CatBoostClassifier()
model.load_model(os.path.join('models', model_name))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = {
            'gender': request.form['gender'].lower(),
            'senior_citizen': request.form['senior_citizen'].lower(),
            'partner': request.form['partner'].lower(),
            'dependents': request.form['dependents'].lower(),
            'tenure': float(request.form['tenure']),
            'phone_service': request.form['phone_service'].lower(),
            'multiple_lines': request.form['multiple_lines'].lower(),
            'internet_service': request.form['internet_service'].lower(),
            'online_security': request.form['online_security'].lower(),
            'online_backup': request.form['online_backup'].lower(),
            'device_protection': request.form['device_protection'].lower(),
            'tech_support': request.form['tech_support'].lower(),
            'streaming_tv': request.form['streaming_tv'].lower(),
            'streaming_movies': request.form['streaming_movies'].lower(),
            'contract': request.form['contract'].lower(),
            'paperless_billing': request.form['paperless_billing'].lower(),
            'payment_method': request.form['payment_method'].lower(),
            'monthly_charges': float(request.form['monthly_charges']),
            'total_charges': float(request.form['total_charges'])
        }
        # Make prediction
        binary_columns = [
            'senior_citizen',
            'partner',
            'dependents',
            'phone_service',
            'paperless_billing'
        ]

        data = []
        for key, value in form_data.items():
            if key == 'gender':
                if value == 'female':
                    data.append(1)
                if value == 'male':
                    data.append(0)
            elif key in binary_columns:
                if value == 'yes':
                    data.append(1)
                if value == 'no':
                    data.append(0)
            else:
                data.append(value)

        probability = round(model.predict_proba([data])[:, -1][0], 4)
        result = round(probability)

        conn = psycopg2.connect(dbname="postgres", user="airflow",
                                password="airflow", host="host.docker.internal"
                                )
        cur = conn.cursor()

        sql = f"""
        INSERT INTO churn_predictions (model_name, prediction, flag)
        VALUES ({model_name}, {probability}, 'web')
        """
        cur.execute(sql)
        conn.commit()

        cur.close()
        conn.close()
        return redirect(url_for('result', result=result,
                                probability=probability))
    return render_template('predict.html')


@app.route('/result')
def result():
    result = request.args.get('result')
    probability = request.args.get('probability')
    background_color = 'green' if int(result) == 1 else 'red'
    return render_template('result.html',
                           probability=probability,
                           result=result,
                           background_color=background_color)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
