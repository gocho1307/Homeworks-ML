import matplotlib.pyplot as plt, pandas as pd, numpy as np
from scipy.io.arff import loadarff
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Read the ARFF file and prepare data
data = loadarff("./data/column_diagnosis.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")
X, y = df.drop("class", axis=1), df["class"]

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=20)
clf.fit(X, y)

# Set style and plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=list(X.columns),
          class_names=list(np.unique(y)), rounded=True, fontsize=12)
plt.savefig("./report/decision_tree.svg")
plt.show()
