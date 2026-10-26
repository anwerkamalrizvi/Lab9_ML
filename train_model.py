from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Train SVM Classifier (a different model than Random Forest)
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Evaluate and save
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained successfully! Accuracy: {acc:.4f}")
joblib.dump(model, 'wine_model.joblib')
