import matplotlib.pyplot as plt, pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Read the ARFF file, prepare data and normalize it
data = loadarff("./data/column_diagnosis.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")
X, y = df.drop("class", axis=1), df["class"]
X_scaled = MinMaxScaler().fit_transform(X)

# Apply PCA to the normalized data
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Convert labels to numerical format
y_numerical = LabelEncoder().fit_transform(y)

# Get k_means with k=3
k_means = KMeans(n_clusters=3, random_state=0)
y_pred = k_means.fit_predict(X_scaled)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the ground diagnoses
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numerical, cmap="viridis")
ax1.set_title("Ground Diagnoses")
ax1.legend(handles=scatter1.legend_elements()[0],
           labels=["Hernia", "Normal", "Spondylolisthesis"])

# Plot the k-means clustering solution (k=3)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="viridis")
ax2.set_title("K-Means Clustering (k=3)")
ax2.legend(handles=scatter2.legend_elements()[0],
           labels=["Cluster 0", "Cluster 1", "Cluster 2"])

plt.show()
