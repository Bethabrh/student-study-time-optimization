import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
# If you don't have real dataset yet, we simulate structure
data = pd.DataFrame({
    "study_time": [2, 3, 5, 1, 4, 6, 2, 7],
    "attendance": [60, 70, 90, 50, 85, 95, 65, 98],
    "assignment_score": [55, 65, 80, 40, 75, 90, 60, 95],
    "performance": [0, 0, 1, 0, 1, 1, 0, 1]  # 0 = low, 1 = high
})

# -----------------------------
# 2. FEATURES & LABEL
# -----------------------------
X = data[["study_time", "attendance", "assignment_score"]]
y = data["performance"]

# -----------------------------
# 3. SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# 4. MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# -----------------------------
# 5. PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 7. SIMPLE STUDY OPTIMIZATION LOGIC
# -----------------------------
def recommend_study_time(study_time, attendance, assignment_score):
    prediction = model.predict([[study_time, attendance, assignment_score]])

    if prediction[0] == 1:
        return "Student is performing well. Maintain current study schedule."
    else:
        return "Increase study time and improve consistency."

# Example test
print(recommend_study_time(3, 65, 60))