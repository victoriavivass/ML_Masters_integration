import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. LOAD DATA
# =============================================================================
_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                    "fr-esr-insertion_professionnelle-master.csv")
masters_data = pd.read_csv(_RAW, sep=";", encoding="utf-8")

# =============================================================================
# 2. FILTER: keep only reliable observations
#    - at least 30 responses
#    - response rate >= 50%
# =============================================================================
masters_data.loc[masters_data["taux_de_reponse"] == "nan", "taux_de_reponse"] = 0
masters_data["taux_de_reponse"] = pd.to_numeric(masters_data["taux_de_reponse"], errors="coerce")

print(len(masters_data))
print(len(masters_data.loc[(masters_data["nombre_de_reponses"] >= 30) & (masters_data["taux_de_reponse"] >= 50)]))

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
        (masters_data_filtered[column] == "nd") | (masters_data_filtered[column] == "ns"), column
    ] = pd.NA
    masters_data_filtered.loc[
        (masters_data_filtered[column] == "nan") | (masters_data_filtered[column] == ".") | (masters_data_filtered[column] == "fe"), column
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
    masters_data_filtered[col] = masters_data_filtered[col].astype(str)
    masters_data_filtered[col] = masters_data_filtered[col].apply(
        lambda x: x.replace("\xa0", "").replace(" ", "")
    )
    masters_data_filtered.loc[
        (masters_data_filtered[col] == "nd") | (masters_data_filtered[col] == "ns"), col
    ] = pd.NA
    masters_data_filtered.loc[
        (masters_data_filtered[col] == "nan") | (masters_data_filtered[col] == ".") |
        (masters_data_filtered[col] == "fe") | (masters_data_filtered[col] == "<NA>"), col
    ] = pd.NA
    masters_data_filtered[col] = pd.to_numeric(masters_data_filtered[col])

# =============================================================================
# 5. DROP columns with more than 70% missing values
# =============================================================================
a_lot_of_missing_values = masters_data_filtered.isnull().mean() * 100
columns_to_keep = a_lot_of_missing_values[a_lot_of_missing_values <= 70].index
masters_data_filtered_less_columns = masters_data_filtered[columns_to_keep]

print(f"Columns removed (>70% missing): {set(masters_data_filtered) - set(masters_data_filtered_less_columns.columns)}")
print(f"Columns kept: {len(masters_data_filtered_less_columns.columns)}")

# =============================================================================
# 6. DROP highly correlated columns
#    salaire_net_median_des_emplois_a_temps_plein  corr > 0.99 with salaire_brut_annuel_estime
#    salaire_net_mensuel_regional_3eme_quartile    corr > 0.99 with salaire_net_mensuel_median_regional
# =============================================================================
masters_data_filtered_less_columns = masters_data_filtered_less_columns.drop(
    "salaire_net_median_des_emplois_a_temps_plein", axis=1
)
masters_data_filtered_less_columns = masters_data_filtered_less_columns.drop(
    "salaire_net_mensuel_regional_3eme_quartile", axis=1
)

# =============================================================================
# 7. VARIABLE NAMES MAPPING (for plots and renamed DataFrames)
# =============================================================================
original_vars = ['annee', 'diplome', 'numero_de_l_etablissement', 'etablissement',
       'etablissementactuel', 'code_de_l_academie', 'academie',
       'code_du_domaine', 'domaine', 'code_de_la_discipline', 'discipline',
       'situation', 'remarque', 'nombre_de_reponses', 'taux_de_reponse',
       'poids_de_la_discipline', 'taux_dinsertion', 'taux_d_emploi',
       'taux_d_emploi_salarie_en_france',
       'emplois_cadre_ou_professions_intermediaires', 'emplois_stables',
       'emplois_a_temps_plein', 'salaire_net_median_des_emplois_a_temps_plein',
       'salaire_brut_annuel_estime', 'de_diplomes_boursiers',
       'taux_de_chomage_regional', 'salaire_net_mensuel_median_regional',
       'emplois_cadre', 'emplois_exterieurs_a_la_region_de_luniversite',
       'femmes', 'salaire_net_mensuel_regional_1er_quartile',
       'salaire_net_mensuel_regional_3eme_quartile', 'size']
new_vars = ['Year', 'Diploma Type', 'Establishment Code', 'Establishment Name',
            'Current Establishment', 'Academy Code', 'Academy Name', 'Department Code',
            'Department', 'Discipline Code', 'Discipline', 'Situation', 'Warnings', 'Number of Responses',
            'Response Rate', 'Sampling Weight', 'Integration Rate', 'Employment Rate',
            'Rate of Paid Employment France', 'Upper/Intermediary Employment Rate',
            'Stable Employment Rate', 'Full Time Employment Rate', 'Median Net Yearly Full Time Salary',
            'Median Net Yearly Salary', 'Scholarship Rate', 'Regional Unemployment Rate', 'Median Regional Monthly Salary',
            'Rate of Upper Occupational Jobs', 'Outside Employment Rate', 'Female Percentage in Program',
            'Monthly Regional Salary 1st Quartile', 'Monthly Regional Salary 3rd Quartile', 'Program Size']

variable_names_mapping = dict(zip(original_vars, new_vars))

# =============================================================================
# 8. CREATE PROXY FOR PROGRAM SIZE
# =============================================================================
masters_data_filtered_less_columns["size"] = (
    masters_data_filtered_less_columns["nombre_de_reponses"] /
    (masters_data_filtered_less_columns["taux_de_reponse"] / 100)
)

# =============================================================================
# 9. BUILD MODELLING DATASETS
#    Drop outcome variables, identifiers, and highly correlated columns
# =============================================================================
# Version using establishment code (numero_de_l_etablissement)
masters_data_filtered_integration_target = masters_data_filtered_less_columns.drop([
    "emplois_cadre_ou_professions_intermediaires", "emplois_stables",
    "emplois_a_temps_plein", "salaire_brut_annuel_estime",
    "cle_etab", "cle_disc", "id_paysage", "emplois_cadre", "taux_de_reponse",
    "nombre_de_reponses", "code_de_la_discipline", "code_du_domaine",
    "code_de_l_academie", "etablissement", "emplois_exterieurs_a_la_region_de_luniversite"
], axis=1)

# Version using establishment name (etablissement)
masters_data_filtered_integration_target_establishment_name = masters_data_filtered_less_columns.drop([
    "emplois_cadre_ou_professions_intermediaires", "emplois_stables",
    "emplois_a_temps_plein", "salaire_brut_annuel_estime",
    "cle_etab", "cle_disc", "id_paysage", "emplois_cadre", "taux_de_reponse",
    "nombre_de_reponses", "code_de_la_discipline", "code_du_domaine",
    "code_de_l_academie", "numero_de_l_etablissement"
], axis=1)

# Drop rows with any remaining NA
masters_data_filtered_integration_target_no_na = masters_data_filtered_integration_target.dropna()
masters_data_filtered_integration_target_establishment_name_no_na = (
    masters_data_filtered_integration_target_establishment_name.dropna()
)

# Rename columns using mapping
masters_data_filtered_integration_target_no_na = (
    masters_data_filtered_integration_target_no_na.rename(columns=variable_names_mapping)
)
masters_data_filtered_integration_target_establishment_name_no_na = (
    masters_data_filtered_integration_target_establishment_name_no_na.rename(columns=variable_names_mapping)
)

print(f"Final modelling dataset shape (establishment code): {masters_data_filtered_integration_target_no_na.shape}")
print(f"Final modelling dataset shape (establishment name): {masters_data_filtered_integration_target_establishment_name_no_na.shape}")
