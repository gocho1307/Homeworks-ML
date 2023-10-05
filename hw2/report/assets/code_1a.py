import matplotlib.pyplot as plt, pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_rel

# Read the ARFF file and prepare data
data = loadarff("./data/column_diagnosis.arff")
df = pd.DataFrame(data[0])
df["class"] = df["class"].str.decode("utf-8")
X, y = df.drop("class", axis=1), df["class"]

# Initialize classifiers
knn_classifier = KNeighborsClassifier(n_neighbors=5)
naive_bayes_classifier = GaussianNB()

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Evaluate classifiers
knn_accuracies = cross_val_score(knn_classifier, X, y, cv=cv, scoring='accuracy')
naive_bayes_accuracies = cross_val_score(naive_bayes_classifier, X, y, cv=cv, scoring='accuracy')

# Plot boxplots
plt.boxplot([knn_accuracies, naive_bayes_accuracies], labels=['kNN', 'Naive Bayes'])
plt.title('Classifier Comparison')
plt.ylabel('Accuracy')
plt.show()