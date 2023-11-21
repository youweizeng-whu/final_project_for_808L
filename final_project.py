from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np



def load_arff_to_dataframe(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    return df, meta


# Function to preprocess the data
def preprocess_data(df, target_column):
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column].astype(str))
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y, le


# Load the dataset
arff_file = 'Rice_Cammeo_Osmancik.arff'
df, metadata = load_arff_to_dataframe(arff_file)

# Preprocess the data
X, y, label_encoder = preprocess_data(df, 'Class')

# Number of folds for cross-validation
n_splits = 10

# Initialize the classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier()
}

# Perform cross-validation and compute confusion matrix
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for name, clf in classifiers.items():
    y_pred = cross_val_predict(clf, X, y, cv=kf)
    conf_matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    print(f"{name} classifier:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n {report}")
    print(f"Confusion Matrix:\n {conf_matrix}\n")

classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier()
}

# Plot ROC Curve
plt.figure(figsize=(10, 8))

for name, clf in classifiers.items():
    # Compute the probability scores
    y_scores = cross_val_predict(clf, X, y, cv=kf, method='predict_proba')

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()