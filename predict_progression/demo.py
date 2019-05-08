from predict_progression.evaluation import plot_evaluation_metrics, \
    model_interpretation, significance_testing

# Step 1. Evaluate the models on 10-fold cross validation.
# cross_validation.compute_evaluation_metrics()

# # Step 2. Plot the results.
plot_evaluation_metrics.scoring_boxplots()
plot_evaluation_metrics.hyperparam_tuning_improvement_boxplots()
plot_evaluation_metrics.performance_heatmap()
#
# # Step 3. Significance testing. Permutation tests for individual classifiers and
# # pairwise Wilcoxon test.
significance_testing.permutation_test_all_models()
significance_testing.wilcoxon_heatmap()

# # Step 4. Compute feature importances using SHAP.
model_interpretation.interpret_shap_values()

# Step 5. Compare with the performance of the ensemble.
plot_evaluation_metrics.per_fold_bar_chart_ensemble()
