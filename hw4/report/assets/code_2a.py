from sklearn.decomposition import PCA

# Apply PCA to the normalized data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Variability explained by the top two principal components
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio for Top 2 PCs: {explained_variance_ratio}")
print(f"Total variability: {explained_variance_ratio[0] + explained_variance_ratio[1]}")