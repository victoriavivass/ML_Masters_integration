import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. LOAD DATA
# =============================================================================
_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                    "fr-esr-insertion_professionnelle-master.csv")
masters_data = pd.read_csv(_RAW, sep=";", encoding="latin-1")

# =============================================================================
# 2. FILTER: keep only reliable observations
#    - at least 30 responses
#    - response rate >= 50%
# =============================================================================
masters_data.loc[masters_data["taux_de_reponse"] == "nan", "taux_de_reponse"] = "0"
masters_data["taux_de_reponse"] = pd.to_numeric(masters_data["taux_de_reponse"], errors="coerce")

masters_data_filtered = masters_data.loc[
    (masters_data["nombre_de_reponses"] >= 30) &
    (masters_data["taux_de_reponse"] >= 50)
].copy()

print(f"Observations after filtering: {len(masters_data_filtered)}")

# =============================================================================
# 3. CLEAN: replace placeholder strings with NA
# =============================================================================
for column in masters_data_filtered.columns:
    masters_data_filtered.loc[
        masters_data_filtered[column].isin(["nd", "ns", "nan", "."]), column
    ] = pd.NA

# =============================================================================
# 4. CONVERT numerical columns to numeric type
# =============================================================================
numerical_cols = [
    "taux_dinsertion",
    "taux_d_emploi",
    "taux_d_emploi_salarie_en_france",
    "emplois_cadre_ou_professions_intermediaires",
    "emplois_stables",
    "emplois_a_temps_plein",
    "salaire_net_median_des_emplois_a_temps_plein",
    "salaire_brut_annuel_estime",
    "de_diplomes_boursiers",
    "taux_de_chomage_regional",
    "salaire_net_mensuel_median_regional",
    "emplois_cadre",
    "emplois_exterieurs_a_la_region_de_luniversite",
    "femmes",
    "salaire_net_mensuel_regional_1er_quartile",
    "salaire_net_mensuel_regional_3eme_quartile",
]

for col in numerical_cols:
    masters_data_filtered[col] = pd.to_numeric(masters_data_filtered[col], errors="coerce")

# =============================================================================
# 5. DROP columns with more than 70% missing values
# =============================================================================
missing_pct = masters_data_filtered.isnull().mean() * 100
columns_to_keep = missing_pct[missing_pct <= 70].index
masters_data_filtered = masters_data_filtered[columns_to_keep]

print(f"Columns kept: {len(masters_data_filtered.columns)}")
print(masters_data_filtered.info())

# =============================================================================
# 6. QUICK EXPLORATION
# =============================================================================

# Correlation matrix (numerical columns only)
numerical_df = masters_data_filtered.select_dtypes(include=["number"]).dropna()

plt.figure(figsize=(12, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

print("\nUnique values per categorical column:")
print(f"  disciplines : {masters_data_filtered['discipline'].nunique()}")
print(f"  domaines    : {masters_data_filtered['code_du_domaine'].nunique()}")
print(f"  diplomes    : {masters_data_filtered['diplome'].nunique()}")
print(f"  annees      : {sorted(masters_data_filtered['annee'].unique())}")
print(f"  universites : {masters_data_filtered['numero_de_l_etablissement'].nunique()}")
print(f"  academies   : {masters_data_filtered['code_de_l_academie'].nunique()}")