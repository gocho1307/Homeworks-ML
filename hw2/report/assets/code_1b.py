from scipy.stats import ttest_rel

# Is kNN better than Naive Bayes?
res = ttest_rel(knn_accs, nb_accs, alternative="greater")
print("Is kNN > Naive Bayes? pval =", res.pvalue)
