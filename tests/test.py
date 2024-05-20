import pandas as pd
import os

import catboost as cat

df = pd.read_csv(os.path.join('..', 'data', 'data.csv'))


def test_numeric():
    assert df['TENURE'].dtype == 'int64'
    assert df['TOTAL_CHARGES'].dtype == 'float64'
    assert df['MONTHLY_CHARGES'].dtype == 'float64'


def test_model():
    model = cat.CatBoostClassifier()
    model.load_model(os.path.join('..', 'models', 'model'))
    form_data = dict(df[model.feature_names_].iloc[0])

    binary_columns = [
        'senior_citizen',
        'partner',
        'dependents',
        'phone_service',
        'paperless_billing'
    ]

    data = []
    for key, value in form_data.items():
        if key.lower() == 'gender':
            if value == 'female':
                data.append(1)
            if value == 'male':
                data.append(0)
        elif key.lower() in binary_columns:
            if value == 'yes':
                data.append(1)
            if value == 'no':
                data.append(0)
        else:
            data.append(value)
    
    pred = model.predict_proba([data])

    assert isinstance(pred[0][0], float)
    assert isinstance(pred[0][1], float)
    assert 0 <= pred[0][0] <= 1
    assert 0 <= pred[0][1] <= 1

