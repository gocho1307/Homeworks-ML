from scipy.stats import ttest_rel

# Is knn better than naive bayes?
res = ttest_rel(knn_accs, nb_accs, alternative="greater")
print("Is knn > naive bayes? pval =", res.pvalue)
