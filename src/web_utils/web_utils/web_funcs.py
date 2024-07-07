from flask import render_template, request, redirect, url_for
import psycopg2


def predict(model, model_name, db_name, 
            db_user, db_pass, db_host,
            db_port):
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

        conn = psycopg2.connect(dbname=db_name, user=db_user,
                                password=db_pass, host=db_host,
                                port=db_port
                                )
        cur = conn.cursor()

        sql = f"""
        INSERT INTO churn_predictions (model_name, prediction, flag)
        VALUES ('{model_name}', {probability}, 'web')
        """
        cur.execute(sql)
        conn.commit()

        cur.close()
        conn.close()
        return redirect(url_for('result_route', result=result,
                                probability=probability))
    return render_template('predict.html')


def result():
    result = request.args.get('result')
    probability = request.args.get('probability')
    background_color = 'green' if int(result) == 1 else 'red'
    return render_template('result.html',
                           probability=probability,
                           result=result,
                           background_color=background_color)
