import numpy as np, matplotlib.pyplot as plt, pandas as pd
from scipy.io.arff import loadarff
from sklearn.feature_selection import f_classif

# Read the ARFF file and prepare data
data = loadarff("./data/column_diagnosis.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")
X, y = df.drop("class", axis=1), df["class"]

# Apply f_classif
f_scores, p_values = f_classif(X, y)

# Obtains the variables with the highest and lowest discriminative power.
highest_discriminative_power_idx = np.argmax(f_scores)
lowest_discriminative_power_idx = np.argmin(f_scores)

highest_discriminative_power_variable = X.columns[
    highest_discriminative_power_idx
]
lowest_discriminative_power_variable = X.columns[
    lowest_discriminative_power_idx
]

# Identifies the input variables requested
print(
    f"Highest discriminative power variable: {highest_discriminative_power_variable}"
)
print(
    f"Lowest discriminative power variable: {lowest_discriminative_power_variable}"
)

plt.figure(figsize=(10, 6))

# Plot for the highest discriminative power variable
for class_label in np.unique(y):
    class_data = X.loc[y == class_label, highest_discriminative_power_variable]
    density, bins = np.histogram(class_data, bins=20, density=True)
    plt.plot(
        bins[:-1],
        density,
        label=f"Class {class_label} - {highest_discriminative_power_variable}",
        linewidth=2,
    )

# Plot for the lowest discriminative power variable
for class_label in np.unique(y):
    class_data = X.loc[y == class_label, lowest_discriminative_power_variable]
    density, bins = np.histogram(class_data, bins=20, density=True)
    plt.plot(
        bins[:-1],
        density,
        linestyle="--",
        label=f"Class {class_label} - {lowest_discriminative_power_variable}",
        linewidth=2,
    )

plt.xlabel("Value")
plt.ylabel("Density")

plt.legend()
plt.grid(True)
plt.savefig("./report/class_conditional_probability.svg")
plt.show()
