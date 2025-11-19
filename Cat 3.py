import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np

# Step 1: Load and anonymize dataset (privacy protection)
df = pd.read_csv("students.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop rows with missing target values
df = df.dropna(subset=['final_result'])

# Ensure target column is integer
df['final_result'] = df['final_result'].astype(int)

# Step 2: Define features and target (relevant features)
X = df.drop("final_result", axis=1)
y = df["final_result"]

numeric_features = ["previous_enrollments","gpa","attendance_rate","failed_courses","age"]
categorical_features = ["program_applied","gender","region"]

# Step 3: Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 4: Build Random Forest pipeline (suitable ML algorithm)
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Step 5: Split dataset (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Evaluate model (communicate results)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("=== Model Evaluation ===")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)
# Step 8: Feature Importance (actionable insights)
# Extract feature importance from Random Forest
rf_model = model.named_steps['rf']
preprocessed_features = (numeric_features + 
                         list(model.named_steps['preprocess']
                              .named_transformers_['cat']
                              .named_steps['onehot']
                              .get_feature_names_out(categorical_features)))

feature_importance = pd.DataFrame({
    'Feature': preprocessed_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Step 9: Student segmentation (actionable communication)
# Get predicted probabilities for enrollment
probabilities = model.predict_proba(X_test)[:,1]

# Segment students
segmentation = pd.DataFrame({
    'Predicted_Probability': probabilities,
    'Predicted_Label': predictions
})
segmentation['Risk_Level'] = pd.cut(segmentation['Predicted_Probability'],
                                    bins=[0, 0.4, 0.7, 1],
                                    labels=['High Risk','Medium Risk','Low Risk'])
print("\nStudent Segmentation:\n", segmentation)
