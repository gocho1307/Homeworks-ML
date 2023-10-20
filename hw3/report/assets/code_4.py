import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Just like in the previous exercise
data = pd.read_csv("./data/winequality-red.csv", sep=";")
X, y = data.drop("quality", axis=1), data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.2, random_state=0)

rmse = []
for rs in range(1, 11):
    # Train the MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation="relu",
                       early_stopping=True, validation_fraction=0.2,
                       random_state=rs)
    mlp.fit(X_train, y_train)

    # Calculate RMSE
    y_pred = mlp.predict(X_test)
    rmse.append(mean_squared_error(y_test, y_pred, squared=False))

# Print the RMSE with early stopping
print(f"RMSE with early stopping and validation_fraction = 0.2: {np.mean(rmse)}")
