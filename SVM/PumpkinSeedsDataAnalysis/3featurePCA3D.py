# Understand Principal Component Analysis with 3 components of features and 3D Visualization
# Refer Pumpkin Seeds Dataset

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

# PCA for features 
pca = PCA(n_components=3)
# pca = PCA(n_components=3).fit(X_scaled)
X_pca = pca.fit_transform(X_scaled)

# PCA component loading - to know the features in a PCA
loadings = pca.components_

loadings_df = pd.DataFrame(
    loadings,
    columns = X.columns, # orginal feature names
    index = [f'PC{i+1}' for i in range(pca.n_components_)]
)
print(loadings_df)

plt.figure(figsize=(12,6))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0)
plt.title('Feature contribution to Principal Components')
plt.show()

# to get top feature per PC
for pc in loadings_df.index:
    print(f"\nTop features for {pc}:")

    # sort by absoulte value and take top 3
    top_features = loadings_df.loc[pc].abs().sort_values(ascending=False).head(3)
    print(top_features)

# visualize PCA in 3D
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
# map class labels to colors
class_names = y.unique()

color_map = sns.color_palette('viridis', len(class_names))

# for each class with different colors

for i, class_name in enumerate(class_names):
    class_mask = (y==class_name)
    ax.scatter(
        X_pca[class_mask, 0],
        X_pca[class_mask, 1],
        X_pca[class_mask, 2],
        c=[color_map[i]],
        label=class_name,
        s=60,
        alpha=0.8,
        depthshade=True
    )

for i, feature in enumerate(X.columns):
    scale_factor = 1.2 
    ax.text(
        loadings[0,i] * scale_factor,
        loadings[1,i] * scale_factor,
        loadings[2,i] * scale_factor,
        feature,
        color='red',
        fontsize=9,
        ha='center',
        va='center'
    )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
ax.set_title('3D PCA Visualization of Pumpkin Seeds Data Red labels show feature contribution')

ax.legend(loc='best')
ax.view_init(elev=25, azim=45)

info_text = f"Explained Variance:\nPC1: {pca.explained_variance_ratio_[0]*100:.1f}%\nPC2: {pca.explained_variance_ratio_[1]*100:.1f}%nPC3: {pca.explained_variance_ratio_[2]*100:.1f}%"
plt.figtext(0.85, 0.15, info_text, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
