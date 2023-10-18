import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Just like in the previous exercise
data = pd.read_csv("./data/winequality-red.csv", sep=";")
X, y = data.drop("quality", axis=1), data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.2, random_state=0)

maes, maes_rb = [], []
for rs in range(1, 11):
    # Train the MLP regressor with a specific number of iterations
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation="relu",
                       early_stopping=True, validation_fraction=0.2,
                       random_state=rs)
    mlp.fit(X_train, y_train)

    # Apply rounding and bounding operations
    y_pred = mlp.predict(X_test)
    y_pred_rb = np.clip(np.round(y_pred), 1, 10)  # Bound between 1 and 10

    # Calculate MAE for both rounded and bounded predictions
    maes.append(mean_absolute_error(y_test, y_pred))
    maes_rb.append(mean_absolute_error(y_test, y_pred_rb))

# Print the MAE for both cases
print(f"MAE without operations: {np.mean(maes)}")
print(f"MAE with rounded and bounded predictions: {np.mean(maes_rb)}")
