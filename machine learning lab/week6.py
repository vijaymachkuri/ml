import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
    data = pd.read_csv(url, sep=',', header=None)
    print(f"Dataset: Length={len(data)}, Shape={data.shape}")
    return data.iloc[:, 1:], data.iloc[:, 0]  # Features (X) and Target (Y)

# Train Decision Tree model
def train_decision_tree(X_train, y_train, criterion="gini"):
    model = DecisionTreeClassifier(criterion=criterion, random_state=100, max_depth=3, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return y_pred

# Plot the decision tree
def plot_tree_model(model, feature_names, class_names):
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()

# Main workflow
if __name__ == "__main__":
    # Load data
    X, Y = load_data()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    
    # Train and evaluate models
    for criterion in ["gini", "entropy"]:
        print(f"\nResults Using {criterion.capitalize()} Index:")
        model = train_decision_tree(X_train, y_train, criterion=criterion)
        evaluate_model(model, X_test, y_test)
        plot_tree_model(model, feature_names=['X1', 'X2', 'X3', 'X4'], class_names=['L', 'B', 'R'])
