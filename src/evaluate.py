"""
evaluate.py — Exploration and diagnostics plots used across the project.

Covers:
  - Correlation matrix
  - Pairplot and binned scatter of response variables
  - Distribution of categorical features
  - Binary-encoded EDA (numerical + categorical conditioned on Low Integration)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_cleaning import (
    masters_data_filtered_less_columns,
    masters_data_filtered_integration_target_no_na,
    variable_names_mapping,
)
from features import numerical_varsb, categorical_varsb

_FIGURES = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(_FIGURES, exist_ok=True)

# =============================================================================
# 1. CORRELATION MATRIX
# =============================================================================
numerical_columns = masters_data_filtered_less_columns.select_dtypes(include=['number'])
correlation_matrix = (
    masters_data_filtered_less_columns.dropna()[numerical_columns.columns]
    .rename(columns=variable_names_mapping)
    .corr()
)

plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of the Data')
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "correlation_matrix.png"), bbox_inches='tight')
plt.show()

# =============================================================================
# 2. PAIRPLOT OF RESPONSE VARIABLES
# =============================================================================
response_vars = [
    "taux_dinsertion",
    "emplois_cadre_ou_professions_intermediaires",
    "emplois_stables",
    "emplois_a_temps_plein",
    "salaire_brut_annuel_estime",
]
# Only available columns
response_vars = [v for v in response_vars if v in masters_data_filtered_less_columns.columns]

sns.pairplot(masters_data_filtered_less_columns[response_vars])
plt.suptitle("Pairplot of Response Features")
plt.savefig(os.path.join(_FIGURES, "pairplot_response.png"), bbox_inches='tight')
plt.show()

# =============================================================================
# 3. BINNED SCATTER: integration rate vs employment quality
# =============================================================================
binned = (
    masters_data_filtered_less_columns
    .groupby('taux_dinsertion', observed=True)
    .agg(
        emplois_cadre_ou_professions_intermediaires=('emplois_cadre_ou_professions_intermediaires', 'mean'),
        emplois_stables=('emplois_stables', 'mean'),
        emplois_a_temps_plein=('emplois_a_temps_plein', 'mean'),
        salaire_brut_annuel_estime=('salaire_brut_annuel_estime', 'mean'),
    )
    .dropna()
    .reset_index()
    .rename(columns=variable_names_mapping)
)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes.flatten()
variables = [
    "Upper/Intermediary Employment Rate",
    "Stable Employment Rate",
    "Full Time Employment Rate",
    "Median Net Yearly Salary",
]
for i, variable in enumerate(variables):
    sns.scatterplot(data=binned, x=variable, y="Integration Rate", ax=axes[i])
    axes[i].set_title(f'Binned Integration Rate against {variable}')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "binned_integration_rate.png"), bbox_inches='tight')
plt.show()

# =============================================================================
# 4. DISTRIBUTION OF CATEGORICAL FEATURES
# =============================================================================
categorical_features     = ["annee", "domaine", "discipline", "situation", "academie", "diplome"]
categorical_features_new = [variable_names_mapping[x] for x in categorical_features]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 15))
axes = axes.flatten()
for i, feature in enumerate(categorical_features_new):
    sns.countplot(
        data=masters_data_filtered_less_columns.rename(columns=variable_names_mapping),
        x=feature, ax=axes[i]
    )
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=27, ha='right')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "categorical_distributions.png"), bbox_inches='tight')
plt.show()

# =============================================================================
# 5. BINARY-ENCODED EDA
#    Numerical and categorical distributions conditioned on Low Integration
# =============================================================================
df_bin = masters_data_filtered_integration_target_no_na.copy()
df_bin["Low Integration"] = 0
df_bin.loc[df_bin["Integration Rate"] <= 85, "Low Integration"] = 1
df_bin = df_bin.drop("Integration Rate", axis=1)

# 5a. KDE plots for numerical variables
n_cols = 2
n_rows = int(np.ceil(len(numerical_varsb) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
axes = axes.flatten()
for i, variable in enumerate(numerical_varsb):
    sns.kdeplot(data=df_bin, x=variable, hue='Low Integration', fill=True, alpha=0.5, ax=axes[i])
    axes[i].set_title(f'Density of {variable}')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "binary_numerical_kde.png"), bbox_inches='tight')
plt.show()

# 5b. Count plots for categorical variables
categorical_varsc = ['Diploma Type', 'Academy Name', 'Department', 'Situation']
n_cols = 2
n_rows = int(np.ceil(len(categorical_varsc) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
axes = axes.flatten()
for i, variable in enumerate(categorical_varsc):
    sns.countplot(data=df_bin, x=variable, ax=axes[i], hue='Low Integration')
    axes[i].set_title(f'Distribution of {variable}')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=40, ha='right')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "binary_categorical_counts.png"), bbox_inches='tight')
plt.show()
