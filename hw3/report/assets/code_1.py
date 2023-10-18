import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Step 1: Load and prepare the dataset
data = pd.read_csv("./data/winequality-red.csv", sep=";")
X, y = data.drop("quality", axis=1), data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.2, random_state=0)

residues = []
for rs in range(1, 11):
    # Step 2: Learn the MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation="relu",
                       early_stopping=True, validation_fraction=0.2,
                       random_state=rs)
    mlp.fit(X_train, y_train)

    # Step 3: Collect the residues
    y_pred = mlp.predict(X_test)
    residues.extend(np.abs(y_pred - y_test))

# Step 4: Plot the distribution of the absolute residues
plt.figure(figsize=(8, 6))
plt.hist(residues, bins=30, color = "#00bfc4", edgecolor="black")
plt.xlabel("Absolute Residues")
plt.ylabel("Count")
plt.show()
