import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, roc_auc_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

from features import (
    preprocessor, preprocessor_dense,
    X, y, w,
    X_d, y_d, w_d,
    X_rf_encoded, y_rf_binary, weights_rf,
    numerical_varsb, categorical_varsb, categorical_vars,
    numerical_vars, variable_names_mapping,
)

_FIGURES = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(_FIGURES, exist_ok=True)

# =============================================================================
# PART 1 — CONTINUOUS REGRESSION
# =============================================================================

# --- Train / test split ---
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.1, random_state=50
)

print("training_expl_shape", X_train.shape)
print("training_resp_shape", y_train.shape)
print("test_expl_shape",     X_test.shape)
print("test_resp_shape",     y_test.shape)

# ---------------------------------------------------------------------------
# 1a. Linear Regression
# ---------------------------------------------------------------------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

scores = cross_val_score(
    pipeline, X_train, y_train, cv=5,
    params={"model__sample_weight": w_train},
    scoring='neg_mean_squared_error'
)
print("Linear Regression — mean MSE:", -scores.mean())
print("Linear Regression — variance:", scores.var())

# ---------------------------------------------------------------------------
# 1b. Ridge Regression
# ---------------------------------------------------------------------------
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge())
])

ridge_param_grid = {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
ridge_param_search = GridSearchCV(
    ridge_pipeline, ridge_param_grid, cv=5, scoring="neg_mean_squared_error"
)
ridge_param_search.fit(X_train, y_train, model__sample_weight=w_train)

ridge_exr_scores = -ridge_param_search.cv_results_['mean_test_score']
print(f"Ridge — overall expected risk: {ridge_exr_scores.mean():.4f}, variance: {ridge_exr_scores.var():.4f}")
print("Ridge — best params:", ridge_param_search.best_params_)

best_param = ridge_param_search.best_index_
print(f"Ridge — best expected risk: {-ridge_param_search.cv_results_['mean_test_score'][best_param]:.4f}")
print(f"Ridge — best variance: {np.square(ridge_param_search.cv_results_['std_test_score'][best_param]):.4f}")

# Ridge shrinkage plot 1 — coefficients > 3 in abs value for all lambda
alphas = np.logspace(-5, 1.3, 100)
coefficients = []
for alpha in alphas:
    ridge_pipeline.set_params(model__alpha=alpha)
    ridge_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    coefficients.append(ridge_pipeline.named_steps['model'].coef_)
coefficients = np.array(coefficients)

mask = np.all(np.abs(coefficients) > 3, axis=0)
preprocessor_fitted = ridge_pipeline.named_steps["preprocessor"]
one_hot_encoder = preprocessor_fitted.named_transformers_["cat"]
categorical_feature_names_out = one_hot_encoder.get_feature_names_out(categorical_varsb)
all_feature_names_for_plot = numerical_varsb + list(categorical_feature_names_out)

plt.figure(figsize=(10, 6))
for i, (name, keep) in enumerate(zip(all_feature_names_for_plot, mask)):
    if keep:
        plt.plot(np.log10(alphas), coefficients[:, i], label=name)
    else:
        plt.plot(np.log10(alphas), coefficients[:, i], color='gray', alpha=0.3)
plt.xlabel('Logarithm of Lambda')
plt.ylabel('Coefficients')
plt.title('Shrinkage of Coefficients with RIDGE')
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "ridge_shrinkage_1.png"), bbox_inches='tight')
plt.show()

# Ridge shrinkage plot 2 — top 12 by ratio high/low alpha
alphas = np.logspace(-5, 7, 100)
coefficients = []
for alpha in alphas:
    ridge_pipeline.set_params(model__alpha=alpha)
    ridge_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    coefficients.append(ridge_pipeline.named_steps['model'].coef_)
coefficients = np.array(coefficients)

coef_at_low_alpha  = np.abs(coefficients[0])
coef_at_high_alpha = np.abs(coefficients[-1])
ratio = abs(coef_at_high_alpha / coef_at_low_alpha)

top_n = 12
top_indices = np.argsort(ratio)[::-1][:top_n]
mask = np.zeros(len(all_feature_names_for_plot), dtype=bool)
mask[top_indices] = True

numerical_feature_names2 = [variable_names_mapping[x] for x in numerical_vars]
categorical_vars2 = [variable_names_mapping[x] for x in categorical_vars]
preprocessor_fitted2 = ridge_pipeline.named_steps["preprocessor"]
one_hot_encoder2 = preprocessor_fitted2.named_transformers_["cat"]
categorical_feature_names_out2 = one_hot_encoder2.get_feature_names_out(categorical_vars2)
all_feature_names_for_plot2 = numerical_feature_names2 + list(categorical_feature_names_out2)

plt.figure(figsize=(10, 6))
for i, (name, highlight) in enumerate(zip(all_feature_names_for_plot2, mask)):
    if highlight:
        plt.plot(np.log10(alphas), coefficients[:, i], label=name, linewidth=2)
    else:
        plt.plot(np.log10(alphas), coefficients[:, i], color='gray', alpha=0.3)
plt.xlabel('Logarithm of Lambda')
plt.ylabel('Coefficients')
plt.title('Shrinkage of Coefficients with RIDGE')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "ridge_shrinkage_2.png"), bbox_inches='tight')
plt.show()

# Ridge shrinkage plot 3 — estimates > 0.005 for all lambda values
coefficients = []
ridge_pipeline.set_params(model__alpha=0.0000000001)
ridge_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
coefficients.append(ridge_pipeline.named_steps['model'].coef_)
print("Minimum absolute coefficient at lowest lambda:", abs(np.array(coefficients)).min())

alphas = np.logspace(-5, 7, 100)
coefficients = []
for alpha in alphas:
    ridge_pipeline.set_params(model__alpha=alpha)
    ridge_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    coefficients.append(ridge_pipeline.named_steps['model'].coef_)
coefficients = np.array(coefficients)
mask = np.all(np.abs(coefficients) > 0.005, axis=0)

plt.figure(figsize=(10, 6))
for i, (name, keep) in enumerate(zip(all_feature_names_for_plot2, mask)):
    if keep:
        plt.plot(np.log10(alphas), coefficients[:, i], label=name)
    else:
        plt.plot(np.log10(alphas), coefficients[:, i], color='gray', alpha=0.3)
plt.xlabel('Logarithm of  Lambda')
plt.ylabel('Coefficients')
plt.title('Shrinkage of Coefficients with RIDGE')
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "ridge_shrinkage_3.png"), bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# 1c. Lasso Regression
# ---------------------------------------------------------------------------
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso())
])

lasso_param_grid = {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
lasso_param_search = GridSearchCV(
    lasso_pipeline, lasso_param_grid, cv=5, scoring="neg_mean_squared_error"
)
lasso_param_search.fit(X_train, y_train, model__sample_weight=w_train)

print("Lasso — best alpha:", lasso_param_search.best_params_)

lasso_exr_scores = -lasso_param_search.cv_results_['mean_test_score']
print(f"Lasso — overall expected risk: {lasso_exr_scores.mean():.4f}, variance: {lasso_exr_scores.var():.4f}")

best_param_lasso = lasso_param_search.best_index_
lasso_mean_best  = -lasso_param_search.cv_results_["mean_test_score"][best_param_lasso]
lasso_var_best   = np.square(lasso_param_search.cv_results_["std_test_score"][best_param_lasso])
print(f"Lasso — best expected risk: {lasso_mean_best:.4f}, best variance: {lasso_var_best:.4f}")

# Lasso shrinkage plot
alphas_lasso = np.arange(0.001, 1, 0.001)
coefficients_lasso = []
for alpha in alphas_lasso:
    lasso_pipeline.set_params(model__alpha=alpha)
    lasso_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    coefficients_lasso.append(lasso_pipeline.named_steps['model'].coef_)
coefficients_lasso = np.array(coefficients_lasso)

coef_at_low_lasso  = np.abs(coefficients_lasso[0])
coef_at_high_lasso = np.abs(coefficients_lasso[-1])
ratio_lasso = abs(coef_at_high_lasso / coef_at_low_lasso)

top_indices_lasso = np.argsort(ratio_lasso)[::-1][:top_n]
mask_lasso = np.zeros(len(all_feature_names_for_plot), dtype=bool)
mask_lasso[top_indices_lasso] = True

preprocessor_lasso = ridge_pipeline.named_steps["preprocessor"]
one_hot_lasso = preprocessor_lasso.named_transformers_["cat"]
cat_names_lasso = one_hot_lasso.get_feature_names_out(categorical_varsb)
all_names_lasso = list(numerical_vars) + list(cat_names_lasso)

plt.figure(figsize=(10, 6))
for i, (name, keep) in enumerate(zip(all_names_lasso, mask_lasso)):
    if keep:
        plt.plot(np.log10(alphas_lasso), coefficients_lasso[:, i], label=name)
    else:
        plt.plot(np.log10(alphas_lasso), coefficients_lasso[:, i], color='gray', alpha=0.3)
plt.xlabel('Logarithm of Alpha')
plt.ylabel('Coefficients')
plt.title('Shrinkage of Coefficients with LASSO')
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "lasso_shrinkage.png"), bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# 1d. Compare all three on test set
# ---------------------------------------------------------------------------
pipeline.fit(X_train, y_train, model__sample_weight=w_train)
y_pred = pipeline.predict(X_test)
print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred):.4f}")

ridge_pipeline.set_params(model__alpha=100)
ridge_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
y_pred_ridge = ridge_pipeline.predict(X_test)
print(f"Ridge Regression MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}")

lasso_pipeline.set_params(model__alpha=0.001)
lasso_pipeline.fit(X_train, y_train, model__sample_weight=w_train)
y_pred_lasso = lasso_pipeline.predict(X_test)
print(f"Lasso Regression MSE: {mean_squared_error(y_test, y_pred_lasso):.4f}")

# Scatter: predicted vs actual
sns.scatterplot(x=y_test, y=y_pred, hue=w_test, palette='viridis')
plt.savefig(os.path.join(_FIGURES, "linear_pred_vs_actual.png"), bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# 1e. Extensions: PLS and Kernel Ridge (from notebook)
# ---------------------------------------------------------------------------
pls_pipeline = Pipeline([
    ('preprocessor', preprocessor_dense),
    ('model', PLSRegression(n_components=20))
])
scores_pls = cross_val_score(pls_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("PLS — mean MSE:", -scores_pls.mean())
print("PLS — variance:", scores_pls.var())

kernel_ridge_pipeline = Pipeline([
    ('scaler', preprocessor),
    ('regressor', KernelRidge())
])
kernel_ridge_param_grid = {
    'regressor__alpha':  [.1, 1, 10, 100],
    'regressor__kernel': ['linear', 'poly', 'rbf'],
    'regressor__gamma':  [0.1, 1, 10, 100]
}
kernel_ridge_grid_search = GridSearchCV(
    kernel_ridge_pipeline, kernel_ridge_param_grid, cv=5, scoring='neg_mean_squared_error'
)
kernel_ridge_grid_search.fit(X_train, y_train)
print("Kernel Ridge — best params:", kernel_ridge_grid_search.best_params_)

best_kr = kernel_ridge_grid_search.best_index_
print("Kernel Ridge — best MSE:", -kernel_ridge_grid_search.cv_results_['mean_test_score'][best_kr])
print("Kernel Ridge — best variance:", kernel_ridge_grid_search.cv_results_['std_test_score'][best_kr]**2)

# =============================================================================
# PART 2 — BINARY CLASSIFICATION: LOGISTIC REGRESSION
# =============================================================================

# --- Train / test split ---
X_d_train, X_d_test, y_d_train, y_d_test, w_d_train, w_d_test = train_test_split(
    X_d, y_d, w_d, test_size=0.1, random_state=42
)

# Check class balance in each split
print("Low integration share — train:", y_d_train.mean().round(3))
print("Low integration share — test: ", y_d_test.mean().round(3))

# ---------------------------------------------------------------------------
# 2a. Simple Logistic Regression (unbalanced)
# ---------------------------------------------------------------------------
LogisticRegression_simple_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

cv_scores = cross_val_score(
    LogisticRegression_simple_pipeline, X_d_train, y_d_train, cv=5, scoring='accuracy'
)
print(f"Logistic (unbalanced) — Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Logistic (unbalanced) — Variance:      {cv_scores.var():.4f}")

LogisticRegression_simple_pipeline.fit(X_d_train, y_d_train)
TN, FP, FN, TP = confusion_matrix(y_d_test, LogisticRegression_simple_pipeline.predict(X_d_test)).ravel()
print("Confusion matrix — TN:", TN, "FP:", FP, "FN:", FN, "TP:", TP)
print(f"Sensitivity: {TP/(TP+FN):.4f}")
print(f"Specificity: {TN/(TN+FP):.4f}")

# ---------------------------------------------------------------------------
# 2b. Balanced Logistic Regression
# ---------------------------------------------------------------------------
LogisticRegression_simple_balanced_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(class_weight="balanced"))
])

cv_scores_bal = cross_val_score(
    LogisticRegression_simple_balanced_pipeline, X_d_train, y_d_train, cv=5, scoring='accuracy'
)
print(f"Logistic (balanced) — Mean Accuracy: {cv_scores_bal.mean():.4f}")
print(f"Logistic (balanced) — Variance:      {cv_scores_bal.var():.4f}")

LogisticRegression_simple_balanced_pipeline.fit(X_d_train, y_d_train)
TN_b, FP_b, FN_b, TP_b = confusion_matrix(
    y_d_test, LogisticRegression_simple_balanced_pipeline.predict(X_d_test)
).ravel()
print("Confusion matrix (balanced) — TN:", TN_b, "FP:", FP_b, "FN:", FN_b, "TP:", TP_b)
print(f"Sensitivity: {TP_b/(TP_b+FN_b):.4f}")
print(f"Specificity: {TN_b/(TN_b+FP_b):.4f}")

# ---------------------------------------------------------------------------
# 2c. Custom threshold search
# ---------------------------------------------------------------------------
model = LogisticRegression_simple_pipeline

def custom_predict(X, threshold):
    predicted_probabilities = model.predict_proba(X)
    return (predicted_probabilities[:, 1] >= threshold).astype(int)

thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
mean_threshold_scores    = []
mean_sensitivity_scores  = []
mean_specificity_scores  = []
variance_threshold_scores   = []
variance_sensitivity_scores = []
variance_specificity_scores = []

for threshold in thresholds:
    list_of_scores    = []
    sensitivity_scores = []
    specificity_scores = []
    for i in range(1, 5):
        LogisticRegression_simple_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression())
        ])
        X_d_train_cv, X_d_test_cv, y_d_train_cv, y_d_test_cv = train_test_split(
            X_d_train, y_d_train, test_size=0.2, random_state=i
        )
        LogisticRegression_simple_pipeline.fit(X_d_train_cv, y_d_train_cv)
        model = LogisticRegression_simple_pipeline
        overall_score     = (custom_predict(X_d_test_cv, threshold) == y_d_test_cv).astype(int).mean()
        sensitivity_score = ((custom_predict(X_d_test_cv, threshold) == 1) & (y_d_test_cv == 1)).astype(int).sum() / (y_d_test_cv == 1).astype(int).sum()
        specificity_score = ((custom_predict(X_d_test_cv, threshold) == 0) & (y_d_test_cv == 0)).astype(int).sum() / (y_d_test_cv == 0).astype(int).sum()
        list_of_scores.append(overall_score)
        sensitivity_scores.append(sensitivity_score)
        specificity_scores.append(specificity_score)
    mean_threshold_scores.append(np.mean(list_of_scores))
    variance_threshold_scores.append(np.var(list_of_scores))
    mean_sensitivity_scores.append(np.mean(sensitivity_scores))
    variance_sensitivity_scores.append(np.var(sensitivity_scores))
    mean_specificity_scores.append(np.mean(specificity_scores))
    variance_specificity_scores.append(np.var(specificity_scores))

print("Threshold mean accuracy scores:   ", mean_threshold_scores)
print("Threshold mean sensitivity scores:", mean_sensitivity_scores)
print("Threshold mean specificity scores:", mean_specificity_scores)

# Best threshold = 0.3
LogisticRegression_simple_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])
LogisticRegression_simple_pipeline.fit(X_d_train, y_d_train)
model = LogisticRegression_simple_pipeline

threshold_best = 0.3
accuracy_03 = (custom_predict(X_d_test, threshold_best) == y_d_test).astype(int).mean()
TN_t, FP_t, FN_t, TP_t = confusion_matrix(y_d_test, custom_predict(X_d_test, threshold_best)).ravel()
print(f"Threshold 0.3 — Accuracy: {accuracy_03:.4f}")
print(f"Threshold 0.3 — Sensitivity: {TP_t/(TP_t+FN_t):.4f}")
print(f"Threshold 0.3 — Specificity: {TN_t/(TN_t+FP_t):.4f}")

# ROC curve
logreg_probs = LogisticRegression_simple_pipeline.predict_proba(X_d_test)
logreg_fpr, logreg_tpr, _ = roc_curve(y_true=y_d_test, y_score=logreg_probs[:, 1])
logreg_auc = roc_auc_score(y_d_test, logreg_probs[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(logreg_fpr, logreg_tpr, label=f'Logistic Regression (AUC = {logreg_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(_FIGURES, "roc_curve.png"), bbox_inches='tight')
plt.show()
print(f"Logistic Regression AUC: {logreg_auc:.2f}")

# =============================================================================
# PART 3 — BINARY CLASSIFICATION: RANDOM FOREST
# =============================================================================
from sklearn.model_selection import train_test_split as tts

X_rf_train, X_rf_test, y_rf_train, y_rf_test, w_rf_train, w_rf_test = tts(
    X_rf_encoded, y_rf_binary, weights_rf,
    test_size=0.2, random_state=42, stratify=y_rf_binary
)

print(f"\ny_train shape: {y_rf_train.shape}")
print(f"y_test shape:  {y_rf_test.shape}")
print(f"Proportion of vulnerable programs in train: {y_rf_train.mean():.3f}")
print(f"Proportion of vulnerable programs in test:  {y_rf_test.mean():.3f}")

rf_pipeline = Pipeline([
    ('rf', RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

param_grid_rf = {
    'rf__n_estimators':    [100, 300],
    'rf__max_depth':       [None, 20],
    'rf__min_samples_split': [2, 10],
    'rf__min_samples_leaf':  [1, 4],
    'rf__max_features':    ['sqrt', None]
}

cv_rf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid_rf,
    cv=cv_rf,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)
grid_search_rf.fit(X_rf_train, y_rf_train, rf__sample_weight=w_rf_train)

best_rf_pipeline = grid_search_rf.best_estimator_
print("\nBest parameters found:")
print(grid_search_rf.best_params_)
print("\nBest CV F1 score:", grid_search_rf.best_score_)

best_rf_pipeline.fit(X_rf_train, y_rf_train, rf__sample_weight=w_rf_train)

y_rf_train_pred  = best_rf_pipeline.predict(X_rf_train)
y_rf_test_pred   = best_rf_pipeline.predict(X_rf_test)
y_rf_train_proba = best_rf_pipeline.predict_proba(X_rf_train)[:, 1]
y_rf_test_proba  = best_rf_pipeline.predict_proba(X_rf_test)[:, 1]

print("\nTraining performance:")
print("Accuracy:",  accuracy_score(y_rf_train, y_rf_train_pred))
print("Precision:", precision_score(y_rf_train, y_rf_train_pred, zero_division=0))
print("Recall:",    recall_score(y_rf_train, y_rf_train_pred, zero_division=0))
print("F1:",        f1_score(y_rf_train, y_rf_train_pred, zero_division=0))
print("ROC-AUC:",   roc_auc_score(y_rf_train, y_rf_train_proba))

print("\nTest performance:")
print("Accuracy:",  accuracy_score(y_rf_test, y_rf_test_pred))
print("Precision:", precision_score(y_rf_test, y_rf_test_pred, zero_division=0))
print("Recall:",    recall_score(y_rf_test, y_rf_test_pred, zero_division=0))
print("F1:",        f1_score(y_rf_test, y_rf_test_pred, zero_division=0))
print("ROC-AUC:",   roc_auc_score(y_rf_test, y_rf_test_proba))

print("\nClassification report on test set:")
print(classification_report(y_rf_test, y_rf_test_pred, zero_division=0))

print("\nConfusion matrix on test set:")
print(confusion_matrix(y_rf_test, y_rf_test_pred))

comparison_df = pd.DataFrame({
    "Actual":    y_rf_test.values,
    "Predicted": y_rf_test_pred,
    "Predicted_Probability_Low_Integration": y_rf_test_proba
})
print("\nFirst 10 predictions on the test set:")
print(comparison_df.head(10))

feature_importance = pd.DataFrame({
    "Feature":    X_rf_train.columns,
    "Importance": best_rf_pipeline.named_steps["rf"].feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\nTop 20 most important features:")
print(feature_importance.head(20))
