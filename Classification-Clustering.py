# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# 2. LOAD KAGGLE DATASET
df = pd.read_csv("/Users/Ishi/Documents/ml/archive/data.csv")

# 3. SELECT NUMERICAL FEATURES
X = df.select_dtypes(include=np.number)

print("Features used for clustering:")
print(X.columns)

print("\nShape of X:", X.shape)

# Replace missing values with the median of each column
X = X.fillna(X.median())

# 4. STANDARDIZE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled data shape:", X_scaled.shape)

# 5. K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=2,init='k-means++',n_init=10,random_state=42)

kmeans_preds = kmeans.fit_predict(X_scaled)

# print("K-Means cluster assignments:")
# print(kmeans_preds)

# 6. HIERARCHICAL AGGLOMERATIVE CLUSTERING
hierarchical = AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')

hc_preds = hierarchical.fit_predict(X_scaled)

# print("Hierarchical cluster assignments:")
# print(hc_preds)

# 7. PCA VISUALIZATION
pca_viz = PCA(n_components=2)

X_2d = pca_viz.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca_viz.explained_variance_ratio_)

print("Total variance explained:",pca_viz.explained_variance_ratio_.sum())

# 8. VISUALIZE CLUSTERS
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means
axes[0].scatter(X_2d[:, 0],X_2d[:, 1],c=kmeans_preds,cmap='viridis',s=40,edgecolor='k')

axes[0].set_title("K-Means Clustering (K=2)")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")


# Hierarchical
axes[1].scatter(X_2d[:, 0],X_2d[:, 1],c=hc_preds,cmap='plasma',s=40,edgecolor='k')

axes[1].set_title("Hierarchical Clustering (Ward's Linkage)")
axes[1].set_xlabel("PC 1")
axes[1].set_ylabel("PC 2")

plt.tight_layout()
plt.show()

# 9. DENDROGRAM
n_samples = min(40, len(X_scaled))

plt.figure(figsize=(12, 5))

linked = linkage(X_scaled[:n_samples],method='ward')

dendrogram(linked,orientation='top',leaf_font_size=10)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Euclidean Distance")

plt.tight_layout()
plt.show()

# CLASSIFICATION - Random Forest
# Classification

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load labels
labels = pd.read_csv("/Users/Ishi/Documents/ml/archive/labels.csv")

# Target = cancer type
y = labels["Class"]

# Features
X = df.drop(columns=["Unnamed: 0"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))