# Understand Principal Component Analysis with 2 components of features and 2D Visualization
# Pumpkin Seeds Dataset

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

# clas data distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Class', data=df)
plt.title('Class distribution')
# plt.show()

# corelation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('feature and corelation matrix')
# plt.show()

X = df.drop('Class', axis=1)
# print(X.head())
y = df['Class']

# standardize feature
sclr = StandardScaler()
X_scaled = sclr.fit_transform(X)

# PCA for features 
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)



# visualize a PCA in 2D
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='viridis')
plt.title('PCA Visualization of Pumpkin Seeds Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Pricipal Component 2')
plt.show()
