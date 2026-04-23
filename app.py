import os
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

def run_model():
    columns = ['age','workclass','fnlwgt','education','education-num',
               'marital-status','occupation','relationship','race','sex',
               'capital-gain','capital-loss','hours-per-week','native-country','income']

    df = pd.read_csv('adult/adult.csv', header=None, names=columns)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.replace('?', pd.NA).dropna()

    le = LabelEncoder()
    df_encoded = df.apply(le.fit_transform)

    X = df_encoded.drop('income', axis=1)
    y = df_encoded['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_before = accuracy_score(y_test, y_pred)
    dp_before = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test['sex'])

    mitigator = ExponentiatedGradient(LogisticRegression(max_iter=1000), DemographicParity())
    mitigator.fit(X_train, y_train, sensitive_features=X_train['sex'])
    y_pred_mitigated = mitigator.predict(X_test)

    accuracy_after = accuracy_score(y_test, y_pred_mitigated)
    dp_after = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=X_test['sex'])

    return {
        'accuracy_before': round(accuracy_before * 100, 2),
        'accuracy_after': round(accuracy_after * 100, 2),
        'bias_before': round(dp_before, 4),
        'bias_after': round(dp_after, 4),
        'bias_reduction': round((dp_before - dp_after) / dp_before * 100, 2)
    }

def get_gemini_explanation(data):
    return (
        f"Before mitigation, the AI model showed a demographic parity bias score of {data['bias_before']}, "
        f"meaning it was making unfair predictions based on gender in hiring decisions. "
        f"After applying Fairlearn's bias mitigation technique, the bias score dropped to {data['bias_after']}, "
        f"a {data['bias_reduction']}% reduction — while maintaining {data['accuracy_after']}% model accuracy. "
        f"This improvement ensures the automated hiring system treats all genders more equally, "
        f"which is critical for fair employment opportunities."
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    data = run_model()
    explanation = get_gemini_explanation(data)
    data['explanation'] = explanation
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)