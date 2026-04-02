import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import (
    masters_data_filtered_integration_target_no_na,
    masters_data_filtered_integration_target_establishment_name_no_na,
    variable_names_mapping,
)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
# Categorical and numerical variable names (original, pre-rename)
categorical_vars  = ["diplome", "numero_de_l_etablissement", "academie", "domaine", "discipline", "situation"]
numerical_vars    = ["taux_de_chomage_regional", "salaire_net_mensuel_median_regional",
                     "femmes", "salaire_net_mensuel_regional_1er_quartile", "size"]

# Renamed versions (used after .rename(columns=variable_names_mapping))
categorical_varsb = [variable_names_mapping[x] for x in categorical_vars]
numerical_varsb   = [variable_names_mapping[x] for x in numerical_vars]

# =============================================================================
# PREPROCESSOR  (shared by all regression and logistic models)
# =============================================================================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                    numerical_varsb),
    ('cat', OneHotEncoder(drop='first'),         categorical_varsb),
])

# Dense version needed for PLS
preprocessor_dense = ColumnTransformer([
    ('num', StandardScaler(),                                            numerical_varsb),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_varsb),
])

# =============================================================================
# CONTINUOUS REGRESSION DATASETS  (X, y, w)
# =============================================================================
df_cont = masters_data_filtered_integration_target_no_na.copy()

X = df_cont.drop(["Integration Rate", "Sampling Weight"], axis=1)
y = df_cont["Integration Rate"]
w = pd.to_numeric(df_cont["Sampling Weight"])

# =============================================================================
# BINARY CLASSIFICATION DATASETS  (X_d, y_d, w_d)
# =============================================================================
df_bin = masters_data_filtered_integration_target_no_na.copy()

# Encode low integration: 1 if Integration Rate <= 85 (25th percentile)
df_bin["Low Integration"] = 0
df_bin.loc[df_bin["Integration Rate"] <= 85, "Low Integration"] = 1

df_bin_target = df_bin.drop("Integration Rate", axis=1)

X_d = df_bin_target.drop(["Low Integration", "Sampling Weight"], axis=1)
y_d = df_bin_target["Low Integration"]
w_d = pd.to_numeric(df_bin_target["Sampling Weight"])

# =============================================================================
# RANDOM FOREST DATASET  (uses establishment name version + get_dummies)
# =============================================================================
rf_categorical_vars = ["diplome", "etablissement", "academie", "domaine", "discipline", "situation"]

df_rf = masters_data_filtered_integration_target_establishment_name_no_na.copy()

# Rename back to original column names expected in the RF notebook
inv_map = {v: k for k, v in variable_names_mapping.items()}
df_rf = df_rf.rename(columns=inv_map)

# Drop unnecessary columns
df_rf = df_rf.drop(columns=["size"], errors="ignore")

y_rf      = df_rf["taux_dinsertion"]
weights_rf = df_rf["poids_de_la_discipline"]
X_rf      = df_rf.drop(columns=["taux_dinsertion", "poids_de_la_discipline"])

# One-hot encode categorical variables
X_rf_encoded = pd.get_dummies(X_rf, columns=rf_categorical_vars, drop_first=True)

# Binary target from 25th percentile
threshold_rf = y_rf.quantile(0.25)
print(f"25th percentile threshold for taux d'insertion: {threshold_rf:.4f}")
y_rf_binary = (y_rf <= threshold_rf).astype(int)

print("\nClass distribution:")
print(y_rf_binary.value_counts(normalize=True))
print(f"Share of low-integration programs: {y_rf_binary.mean():.3f}")
