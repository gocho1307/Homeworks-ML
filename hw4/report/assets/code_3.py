import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder

# Apply PCA to the normalized data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Convert labels to numerical format
le = LabelEncoder()
y_numerical = le.fit_transform(y)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the ground diagnoses
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numerical, cmap='viridis')
ax1.set_title('Ground Diagnoses')
ax1.legend(handles=scatter1.legend_elements()[0], labels=list(set(y)))

# Plot the k-means clustering solution (k=3)
k_means = cluster.KMeans(n_clusters=3, random_state=0)
y_pred = k_means.fit_predict(X_scaled)

scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
ax2.set_title('K-Means Clustering (k=3)')
ax2.legend(handles=scatter2.legend_elements()[0], labels=['Cluster 0', 'Cluster 1', 'Cluster 2'])

plt.show()