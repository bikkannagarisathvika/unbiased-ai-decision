import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score

# Load data
columns = ['age','workclass','fnlwgt','education','education-num',
           'marital-status','occupation','relationship','race','sex',
           'capital-gain','capital-loss','hours-per-week','native-country','income']

df = pd.read_csv('adult/adult.csv', header=None, names=columns)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Strip whitespace from string values
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Drop missing values
df = df.replace('?', pd.NA).dropna()

# Encode categorical columns
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# Split features and target
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy before
accuracy_before = accuracy_score(y_test, y_pred)

# Sensitive feature - gender
sensitive_feature = X_test['sex']

# Bias before
dp_before = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature)

print(f"Accuracy Before: {accuracy_before}")
print(f"Bias Before (Demographic Parity): {dp_before}")

# Apply mitigation
mitigator = ExponentiatedGradient(LogisticRegression(max_iter=1000), DemographicParity())
mitigator.fit(X_train, y_train, sensitive_features=X_train['sex'])

# New predictions
y_pred_mitigated = mitigator.predict(X_test)

# Accuracy after
accuracy_after = accuracy_score(y_test, y_pred_mitigated)

# Bias after
dp_after = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sensitive_feature)

print(f"Accuracy Before Mitigation: {accuracy_before}")
print(f"Bias Before Mitigation (Demographic Parity): {dp_before}")
print(f"Accuracy After Mitigation: {accuracy_after}")
print(f"Bias After Mitigation (Demographic Parity): {dp_after}")