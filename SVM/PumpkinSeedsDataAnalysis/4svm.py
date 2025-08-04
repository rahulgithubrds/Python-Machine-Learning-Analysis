# Classification by SVM Confusion Matrix and Accuracy for Pumpkin Seeds paramters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import joblib
import os

def load_cached_data():
    if os.path.exists(CACHE_FILE):
        try:
            df = joblib.load(CACHE_FILE)
            print("Successfu1lly loaded cached dataframe")
            return df
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    else:
        print("Cache file not found")
        return None

CACHE_DIR = "data_cache"
CACHE_FILE = os.path.join(CACHE_DIR,"pumpkin_seeds.pkl")

df = load_cached_data()

# print(df.head())

le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

X = df.drop('Class', axis=1)
# print(X.head())
y = df['Class']

# standardize feature
sclr = StandardScaler()
X_scaled = sclr.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print("Training SVM...")
svm = SVC(kernel='rbf', gamma=0.1, C=1, probability=True)
# svm = SVC(kernel='linear', C=1000)
# svm = SVC(kernel='poly', degree=3, C=1)

svm.fit(X_train, y_train)

# Evaluation
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBasic SVM Accuracy: {accuracy:.4f}")

# 8. Confusion Matrix Visualization (Matplotlib only)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()

tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# add count annotations
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black" 
                 )

plt.show()
