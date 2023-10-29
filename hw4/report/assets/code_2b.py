# Get the absolute weights (loadings) of the top two principal components
pc_weights = np.abs(pca.components_[:2, :])

# Get the feature names
feature_names = X.columns

# Sort the feature names by relevance for each PC
sorted_features_pc1 = [feature_names[i] for i in np.argsort(pc_weights[0])[::-1]]
sorted_features_pc2 = [feature_names[i] for i in np.argsort(pc_weights[1])[::-1]]

print(f"Top Variables for PC1: {sorted_features_pc1}")
print(f"Top Variables for PC2: {sorted_features_pc2}")
