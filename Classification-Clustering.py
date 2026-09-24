# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report

from scipy.cluster.hierarchy import linkage, dendrogram

from minisom import MiniSom

# 2. LOAD DATASET
df = pd.read_csv("/Users/Ishi/Documents/ml/archive/data.csv")
labels = pd.read_csv("/Users/Ishi/Documents/ml/archive/labels.csv")

print("Dataset shape:", df.shape)
print("Labels shape:", labels.shape)

# 3. PREPARE FEATURES
# Remove sample ID column
X = df.drop(columns=["Unnamed: 0"])

# Select only numerical gene-expression features
X = X.select_dtypes(include=np.number)

# Handle missing values
X = X.fillna(X.median())

print("Feature shape:", X.shape)

# 4. TARGET VARIABLE
y = labels["Class"]

print("\nCancer classes:")
print(y.value_counts())

# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y)

# 6. STANDARDIZATION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:", X_scaled.shape)

# CLASSIFICATION ALGORITHMS
# 7. LOGISTIC REGRESSION
logistic_model = LogisticRegression(max_iter=1000,random_state=42)
logistic_model.fit(X_train_scaled, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)

print("Accuracy:",accuracy_score(y_test, logistic_pred))

print(classification_report(y_test, logistic_pred))

# 8. SUPPORT VECTOR MACHINE (SVM)
svm_model = SVC(kernel="linear",random_state=42)
svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, svm_pred))

print(classification_report(y_test, svm_pred))

# 9. K-NEAREST NEIGHBORS (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

print("Accuracy:",accuracy_score(y_test, knn_pred))

print(classification_report(y_test, knn_pred))

# 10. RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

# 11. GAUSSIAN NAIVE BAYES
nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)

nb_pred = nb_model.predict(X_test_scaled)

print("Accuracy:",accuracy_score(y_test, nb_pred))

print(classification_report(y_test, nb_pred))

# 12. DECISION TREE
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test, dt_pred))

print(classification_report(y_test, dt_pred))

# CLUSTERING ALGORITHMS
# 13. K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=5,n_init=10,random_state=42)
kmeans_preds = kmeans.fit_predict(X_scaled)

print("K-Means cluster labels:")
print(kmeans_preds)

# 14. HIERARCHICAL CLUSTERING
hierarchical = AgglomerativeClustering(n_clusters=5,metric="euclidean",linkage="ward")
hc_preds = hierarchical.fit_predict(X_scaled)

print("Hierarchical cluster labels:")
print(hc_preds)

# 15. PRINCIPAL COMPONENT ANALYSIS (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA shape:", X_pca.shape)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print("Total variance explained:",
      pca.explained_variance_ratio_.sum())

# 16. PCA VISUALIZATION
plt.figure(figsize=(8, 6))

plt.scatter(X_pca[:, 0],X_pca[:, 1],c=kmeans_preds,s=40)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization with K-Means Clusters")

plt.show()

# 17. SINGULAR VALUE DECOMPOSITION (SVD)
svd = TruncatedSVD(n_components=2,random_state=42)

X_svd = svd.fit_transform(X_scaled)

print("SVD shape:", X_svd.shape)

print("Explained variance ratio:")
print(svd.explained_variance_ratio_)

print("Total variance explained:",
      svd.explained_variance_ratio_.sum())

# 18. SVD VISUALIZATION
plt.figure(figsize=(8, 6))
plt.scatter(X_svd[:, 0],X_svd[:, 1],c=kmeans_preds,s=40)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("SVD Visualization")

plt.show()

# 19. HIERARCHICAL CLUSTERING DENDROGRAM
n_samples = min(40, len(X_scaled))
linked = linkage(X_scaled[:n_samples],method="ward")
plt.figure(figsize=(12, 5))
dendrogram(linked,orientation="top",leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

# 20. SELF-ORGANIZING MAP (SOM)
som = MiniSom(x=7,y=7,input_len=X_scaled.shape[1],sigma=1.0,
              learning_rate=0.5,random_seed=42)
# Initialize SOM weights
som.random_weights_init(X_scaled)
# Train SOM
som.train_random(X_scaled,1000)
print("SOM training completed.")

# 21. SOM VISUALIZATION
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T)
plt.colorbar()
plt.title("Self-Organizing Map")
plt.show()

# 22. CLASSIFICATION RESULTS COMPARISON
results = pd.DataFrame({
    "Algorithm": ["Logistic Regression",
        "SVM",
        "KNN",
        "Random Forest",
        "Gaussian Naive Bayes",
        "Decision Tree"],

    "Accuracy": [accuracy_score(y_test, logistic_pred),
        accuracy_score(y_test, svm_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, dt_pred)]
})
print(results)
# 23. CLASSIFICATION ACCURACY PLOT
plt.figure(figsize=(10, 6))
plt.bar(results["Algorithm"],results["Accuracy"])
plt.xticks(rotation=45,ha="right")
plt.ylabel("Accuracy")
plt.xlabel("Algorithm")
plt.title("Comparison of Classification Algorithms")
plt.tight_layout()
plt.show()

