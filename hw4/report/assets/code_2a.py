import numpy as np, pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Read the ARFF file, prepare data and normalize it
data = loadarff("./data/column_diagnosis.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")
X, y = df.drop("class", axis=1), df["class"]
X_scaled = MinMaxScaler().fit_transform(X)

# Apply PCA to the normalized data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Variability explained by the top two principal components
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio for Top 2 PCs: {explained_variance_ratio}")
print(f"Total variability: {explained_variance_ratio[0] + explained_variance_ratio[1]}")
