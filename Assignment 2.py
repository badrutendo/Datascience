import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

#Load CSV
df = pd.read_csv("road_accidents.csv")

#Display dataset preview
print("Dataset preview:")
print(df.head())

#Define independent (X) and dependent (y) variables
X = df[['Vehicle_Speed', 'Weather_Condition', 'Road_Surface', 'Driver_Age', 'Light_Condition']]
y = df['Accident_Severity']

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Evaluate the model
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("R² Score:", round(r2_score(y_test, y_pred), 3))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 3))

#Save the model
joblib.dump(model, "accident_severity_model.pkl")
print("\nModel saved successfully as 'accident_severity_model.pkl'")

#Predict a new case
new_data = pd.DataFrame({
    'Vehicle_Speed': [100],
    'Weather_Condition': [1],
    'Road_Surface': [1],
    'Driver_Age': [30],
    'Light_Condition': [1]
})

predicted_severity = model.predict(new_data)
print("\nPredicted Accident Severity:", round(predicted_severity[0], 2))
