# Perform paired t-test
t_statistic, p_value = ttest_rel(knn_accuracies, naive_bayes_accuracies)

# Check the p-value
if p_value < 0.05:
    print("Reject null hypothesis: kNN is statistically superior to Naive Bayes in terms of accuracy")
else:
    print("Fail to reject null hypothesis: No significant difference in accuracy between kNN and Naive Bayes")