import os
import requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score
import io

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

def run_model(df, sensitive_feature=None):
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.replace('?', pd.NA).dropna()

    le = LabelEncoder()
    df_encoded = df.apply(le.fit_transform)

    target = df_encoded.columns[-1]
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    sensitive_col = (
        sensitive_feature
        if sensitive_feature and sensitive_feature in X.columns
        else (X.columns[7] if len(X.columns) > 7 else X.columns[0])
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_before = accuracy_score(y_test, y_pred)
    dp_before = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test[sensitive_col])

    pred_series = pd.Series(y_pred, index=X_test.index)
    sensitive_series = X_test[sensitive_col]
    groups = sensitive_series.unique()
    rates = {g: pred_series[sensitive_series == g].mean() for g in groups}
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    di_before = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0

    mitigator = ExponentiatedGradient(LogisticRegression(max_iter=1000), DemographicParity())
    mitigator.fit(X_train, y_train, sensitive_features=X_train[sensitive_col])
    y_pred_mitigated = mitigator.predict(X_test)

    accuracy_after = accuracy_score(y_test, y_pred_mitigated)
    dp_after = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=X_test[sensitive_col])

    pred_mit_series = pd.Series(y_pred_mitigated, index=X_test.index)
    rates_after = {g: pred_mit_series[sensitive_series == g].mean() for g in groups}
    min_rate_after = min(rates_after.values())
    max_rate_after = max(rates_after.values())
    di_after = round(min_rate_after / max_rate_after, 4) if max_rate_after > 0 else 1.0

    return {
        'accuracy_before': round(accuracy_before * 100, 2),
        'accuracy_after': round(accuracy_after * 100, 2),
        'bias_before': round(abs(dp_before), 4),
        'bias_after': round(abs(dp_after), 4),
        'bias_reduction': round((abs(dp_before) - abs(dp_after)) / abs(dp_before) * 100, 2),
        'di_before': di_before,
        'di_after': di_after,
        'sensitive_feature': sensitive_col,
        'total_records': len(df)
    }

def get_gemini_explanation(data):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""You are an AI fairness expert reviewing a bias audit report. Based on the results below, do exactly 4 things:

1. WHAT THE BIAS MEANS: Explain the demographic parity score and disparate impact score in plain English - what do these numbers mean for real people?
2. ROOT CAUSE: What likely caused this bias in the data? Name specific types of features (e.g. historical hiring patterns, education access gaps, occupational segregation) that could explain bias in the '{data['sensitive_feature']}' column.
3. IMPACT OF MITIGATION: Why does the {data['bias_reduction']}% bias reduction matter? Give a concrete real-world example of who benefits.
4. RECOMMENDATIONS: Give 2 specific actions the organization should take next to further reduce bias in their AI system.

Audit Results:
- Dataset size: {data['total_records']} records
- Sensitive feature analyzed: {data['sensitive_feature']}
- Demographic Parity Before: {data['bias_before']} -> After: {data['bias_after']}
- Disparate Impact Before: {data['di_before']} -> After: {data['di_after']} (1.0 = perfectly fair)
- Bias Reduced By: {data['bias_reduction']}%
- Model Accuracy Before: {data['accuracy_before']}% -> After: {data['accuracy_after']}%

Format your response with clear labels: WHAT THE BIAS MEANS, ROOT CAUSE, IMPACT, RECOMMENDATIONS. Keep each section to 2-3 sentences. No jargon."""
            }]
        }]
    }
    response = requests.post(url, json=payload)
    result = response.json()
    if 'candidates' in result:
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return (
            f"WHAT THE BIAS MEANS: The model showed a demographic parity score of {data['bias_before']} "
            f"and disparate impact of {data['di_before']} on '{data['sensitive_feature']}', meaning different groups were treated unequally. "
            f"ROOT CAUSE: Historical patterns in the training data likely reflected existing societal inequalities. "
            f"IMPACT: After mitigation, bias dropped by {data['bias_reduction']}% while maintaining {data['accuracy_after']}% accuracy. "
            f"RECOMMENDATIONS: Regularly audit model outputs and collect more representative training data."
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/columns', methods=['POST'])
def columns():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'columns': [
            'age','workclass','fnlwgt','education','education-num',
            'marital-status','occupation','relationship','race','sex',
            'capital-gain','capital-loss','hours-per-week','native-country'
        ]})
    file = request.files['file']
    df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
    cols = list(df.columns[:-1])
    return jsonify({'columns': cols})

@app.route('/results', methods=['POST'])
def results():
    if 'file' not in request.files or request.files['file'].filename == '':
        df = pd.read_csv('adult/adult.csv', header=None, names=[
            'age','workclass','fnlwgt','education','education-num',
            'marital-status','occupation','relationship','race','sex',
            'capital-gain','capital-loss','hours-per-week','native-country','income'
        ])
    else:
        file = request.files['file']
        df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))

    sensitive_feature = request.form.get('sensitive_feature', '')
    try:
        data = run_model(df, sensitive_feature)
    except Exception as e:
        return jsonify({'error': f'Could not process this dataset. Please upload a classification CSV (hiring, loan, income data) with a binary outcome column. Details: {str(e)}'}), 400

    explanation = get_gemini_explanation(data)
    data['explanation'] = explanation
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)