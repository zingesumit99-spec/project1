"""
Iris Flower Classification using Scikit-learn
================================================
This script demonstrates a complete machine learning workflow:
1. Load and explore the dataset
2. Split data into training and test sets
3. Train multiple classification models
4. Evaluate performance using accuracy, confusion matrix, and classification report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------------
df = pd.read_csv('/mnt/user-data/uploads/Iris.csv')

# Drop the Id column (not a useful feature)
df = df.drop(columns=['Id'])

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution:\n", df['Species'].value_counts())
print("\nSummary statistics:\n", df.describe())

# ----------------------------------------------------------------
# 2. Prepare features (X) and target (y)
# ----------------------------------------------------------------
X = df.drop(columns=['Species'])
y = df['Species']

feature_names = X.columns.tolist()
class_names = sorted(y.unique())

# ----------------------------------------------------------------
# 3. Train/test split (stratified to preserve class balance)
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ----------------------------------------------------------------
# 4. Feature scaling (helps KNN, SVM, Logistic Regression)
# ----------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------
# 5. Train and evaluate multiple models
# ----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='linear')
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------
# 6. Compare model accuracies
# ----------------------------------------------------------------
print("\n" + "="*50)
print("MODEL ACCURACY COMPARISON")
print("="*50)
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name:25s}: {acc:.4f}")

best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")

# ----------------------------------------------------------------
# 7. Detailed look at the best model: Confusion Matrix
# ----------------------------------------------------------------
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best, labels=class_names)

print(f"\nConfusion Matrix for {best_model_name}:")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

# ----------------------------------------------------------------
# 8. Visualizations
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (a) Pairplot-style scatter: Sepal dimensions
ax = axes[0, 0]
for species in class_names:
    subset = df[df['Species'] == species]
    ax.scatter(subset['SepalLengthCm'], subset['SepalWidthCm'], label=species, alpha=0.7)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_title('Sepal Dimensions by Species')
ax.legend()

# (b) Petal dimensions
ax = axes[0, 1]
for species in class_names:
    subset = df[df['Species'] == species]
    ax.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], label=species, alpha=0.7)
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Petal Dimensions by Species')
ax.legend()

# (c) Model accuracy comparison bar chart
ax = axes[1, 0]
names = list(results.keys())
accs = list(results.values())
bars = ax.bar(names, accs, color='steelblue')
ax.set_ylim(0.8, 1.02)
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.tick_params(axis='x', rotation=30)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{acc:.3f}', ha='center', fontsize=9)

# (d) Confusion matrix heatmap for best model
ax = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix: {best_model_name}')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/iris_analysis.png', dpi=120, bbox_inches='tight')
print("\nVisualization saved as iris_analysis.png")

# ----------------------------------------------------------------
# 9. Feature importance (Random Forest)
# ----------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances)

# ----------------------------------------------------------------
# 10. Try a prediction on a new sample
# ----------------------------------------------------------------
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)
print(f"\nSample prediction for {sample.values.tolist()[0]}: {prediction[0]}")
