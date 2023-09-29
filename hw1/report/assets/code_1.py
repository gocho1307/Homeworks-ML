import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.feature_selection import f_classif

# Read the ARFF file and prepare data
data = loadarff("./data/column_diagnosis.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")
X, y = df.drop("class", axis=1), df["class"]

# Apply f_classif
f_scores, _ = f_classif(X, y)

# Obtains the variables with the highest and lowest discriminative power.
h_disc_power_var = X.columns[np.argmax(f_scores)]
l_disc_power_var = X.columns[np.argmin(f_scores)]

plt.figure(figsize=(8, 6))

# Plot for the highest discriminative power variable
for class_label in np.unique(y):
    class_data = X.loc[y == class_label, h_disc_power_var]
    sns.kdeplot(
        class_data,
        label=f"Class {class_label} - {h_disc_power_var}",
        linewidth=2,
    )

# Plot for the lowest discriminative power variable
for class_label in np.unique(y):
    class_data = X.loc[y == class_label, l_disc_power_var]
    sns.kdeplot(
        class_data,
        label=f"Class {class_label} - {l_disc_power_var}",
        linestyle="--",
        linewidth=2,
    )

plt.xlabel("Variables")
plt.ylabel("Density")

plt.legend()
plt.grid(True)
plt.show()
