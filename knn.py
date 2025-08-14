from data_prep import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def run():
    X_train, X_test, y_train, y_test, before, after = load_data()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print("=== K-Nearest Neighbors (k=5) ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("Confusion Matrix:")
    print(cm)
    joblib.dump(model, "knn_k5.pkl")

if __name__ == '__main__':
    run()
