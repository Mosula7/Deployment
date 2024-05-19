import catboost as cat
import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
model = cat.CatBoostClassifier()
model.load_model(os.path.join('models', 'model'))


def make_prediction(data):
    result = model.predict_proba(data)[:, -1]
    return result


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = {
            'gender': request.form['gender'],
            'senior_citizen': request.form['senior_citizen'],
            'partner': request.form['partner'],
            'dependents': request.form['dependents'],
            'tenure': request.form['tenure'],
            'phone_service': request.form['phone_service'],
            'multiple_lines': request.form['multiple_lines'],
            'internet_service': request.form['internet_service'],
            'online_security': request.form['online_security'],
            'online_backup': request.form['online_backup'],
            'device_protection': request.form['device_protection'],
            'tech_support': request.form['tech_support'],
            'streaming_tv': request.form['streaming_tv'],
            'streaming_movies': request.form['streaming_movies'],
            'contract': request.form['contract'],
            'paperless_billing': request.form['paperless_billing'],
            'payment_method': request.form['payment_method'],
            'monthly_charges': request.form['monthly_charges'],
            'total_charges': request.form['total_charges']
        }
        # Make prediction
        result = make_prediction(list(form_data.values()))
        return redirect(url_for('result', result=result))
    return render_template('predict.html')


@app.route('/result')
def result():
    result = request.args.get('result')
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
