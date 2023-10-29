import numpy as np, pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn import cluster, metrics

# Read the ARFF file, prepare data and normalize it
data = loadarff("./data/column_diagnosis.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")
X, y = df.drop("class", axis=1), df["class"]
X_scaled = MinMaxScaler().fit_transform(X)

# Parametrize the clustering and learn the model
k_means_models = []
for n_clusters in [2, 3, 4, 5]:
    k_means = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    k_means_models.append(k_means.fit(X_scaled))

for model in k_means_models:
    n_clusters = model.n_clusters
    y_pred = model.labels_

    # Calculate the silhouette
    silhouette = metrics.silhouette_score(X_scaled, y_pred)

    # Calculate the purity
    conf_matrix = metrics.cluster.contingency_matrix(y, y_pred)
    purity = np.sum(np.amax(conf_matrix, axis=0)) / np.sum(conf_matrix)

    # Print the results for each number of clusters
    print(f"Clustering with n_clusters = {n_clusters}")
    print(f"\tSilhouette = {silhouette:6.5f}")
    print(f"\tPurity = {purity:6.5f}")
    print()
