# Imperial College course mapping

This mapping is based on the connected course notebook and its listed course sources. It uses the
same conventional variable notation in code samples.

| Course concept | BenignIDS v4 demonstration |
|---|---|
| Python, NumPy, pandas | Chunk-aware data loading, vectorized token arrays, comparison tables |
| Probability and statistics | P(y = 1 "|" X)`, calibration/Brier score, uncertainty and label noise |
| Training, validation, test sets | Three-way stratified split with the test set opened once |
| Cross-validation | `StratifiedKFold` optimizing average precision |
| Preprocessing | Median imputation, scaling, one-hot encoding, byte tokenization |
| PCA/dimensionality reduction | Optional notebook comparison for linear models and visualization |
| Bias–variance and overfitting | learning curves, early stopping, regularization, bagging |
| Imbalanced learning | class weights, PR curves, PR-AUC, precision-constrained thresholds |
| Neural networks | 1D-CNN benchmark and transformer encoder |
| Model tuning | manual/default, grid, randomized, and Bayesian searches |
| Bayesian optimization | Gaussian-process surrogate and acquisition-driven candidate selection |
| Ensembles | random-forest bagging, probability voting, logistic stacking |
| Reinforcement learning/bandits | acquisition exploration–exploitation analogy, not an RL claim |
| Monte Carlo reasoning | repeated masks/noise seeds and empirical robustness intervals |
| Responsible AI | provenance, test isolation, limitations, model card, SHAP |

## Mathematical notation

The classifier estimates

`y_score = P(y = 1 | X)`

and converts probabilities into decisions using

`y_hat = 1[y_score >= tau]`.

`tau` is selected on `X_val`, never on `X_test`. For rare threats, the central comparison is the
precision–recall curve and its area (average precision), not raw accuracy.

Bayesian optimization chooses the next hyperparameter vector `theta` by maximizing an acquisition
function `a(theta)`. The surrogate's uncertainty supports exploration, while its predicted high
score supports exploitation. This is analogous to a bandit trade-off but is not itself a learned
SOC response policy.

