# SmartStreet Demand Prediction Project
# Author: Vrinda Sharma
# Purpose: Predict street food demand levels using ML models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings

warnings.filterwarnings("ignore")

# ------------------ Load Dataset ------------------
df = pd.read_csv("indian_food.csv")
print("✅ Dataset Loaded Successfully!\n")
print(df.head(), "\n")

# ------------------ Data Cleaning ------------------
df.replace("-1", np.nan, inplace=True)
df.dropna(subset=['prep_time', 'cook_time', 'flavor_profile', 'course', 'region'], inplace=True)
df['prep_time'] = df['prep_time'].astype(int)
df['cook_time'] = df['cook_time'].astype(int)

# ------------------ Feature Engineering ------------------
df['total_time'] = df['prep_time'] + df['cook_time']
df['num_ingredients'] = df['ingredients'].apply(lambda x: len(str(x).split(',')))

def assign_demand(row):
    if row['course'] == 'snack' and row['total_time'] <= 30:
        return 'High'
    elif row['course'] == 'main course' and row['total_time'] <= 60:
        return 'Medium'
    else:
        return 'Low'

df['demand_level'] = df.apply(assign_demand, axis=1)

# ------------------ Encoding ------------------
cat_cols = ['diet', 'flavor_profile', 'course', 'region']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df[['diet', 'flavor_profile', 'course', 'region', 'total_time', 'num_ingredients']]
y = le.fit_transform(df['demand_level'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ Model Training ------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    conf_matrices[name] = confusion_matrix(y_test, preds)
    print(f"\n🔍 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

# ------------------ ANN Model ------------------
ann = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

loss, acc = ann.evaluate(X_test, y_test, verbose=0)
results["ANN (Keras)"] = acc
conf_matrices["ANN (Keras)"] = confusion_matrix(y_test, np.argmax(ann.predict(X_test), axis=1))

# ------------------ Sheet 1: EDA Visualizations ------------------
plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(x='demand_level', data=df, ax=axes[0, 0], palette="mako")
axes[0, 0].set_title("Demand Level Distribution")

sns.heatmap(df[['prep_time', 'cook_time', 'total_time', 'num_ingredients']].corr(), annot=True, cmap="coolwarm", ax=axes[0, 1])
axes[0, 1].set_title("Correlation Heatmap")

sns.boxplot(x='demand_level', y='total_time', data=df, ax=axes[1, 0], palette="cool")
axes[1, 0].set_title("Total Time vs Demand")

sns.scatterplot(x='num_ingredients', y='total_time', hue='demand_level', data=df, ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title("Ingredients vs Total Time")

plt.tight_layout()
plt.show()

# ------------------ Sheet 2: Accuracy Comparison ------------------
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="crest")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ------------------ Sheet 3: Confusion Matrices ------------------
num_models = len(conf_matrices)
rows = (num_models + 1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))
axes = axes.flatten()

for i, (name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=axes[i])
    axes[i].set_title(f"Confusion Matrix - {name}", fontsize=12)
    axes[i].set_xlabel("Predicted", fontsize=10)
    axes[i].set_ylabel("Actual", fontsize=10)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3.0)
plt.show()

# ------------------ Final Accuracy Summary ------------------
print("\n--- MODEL ACCURACY SUMMARY ---")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")
print("----------------------------------")
print("✅ All models trained successfully! Use the highest-accuracy model for deployment.")