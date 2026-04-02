# Predicting Professional Integration After French Master's Programs

**Authors:** Célina Madaschi · Ana Catarina da Fonseca Gaspar · María Victoria Vivas Gutiérrez

---

## Objective

Predict how well-integrated alumni from French Master's programs are into the job market. Two complementary framings are used:

- **Continuous prediction** — predict the exact insertion rate using Linear, Ridge, and Lasso regression.
- **Binary classification** — predict whether a program is *low integration* (insertion rate ≤ 85th–86th percentile, i.e., below the P25 threshold) using Logistic Regression and Random Forest.

---

## Dataset

| Field | Value |
|---|---|
| **Name** | Insertion professionnelle des diplômés de Master en universités et établissements assimilés |
| **Source** | Ministère de l'Enseignement supérieur et de la Recherche (MESR-SIES) |
| **Licence** | Open Data — Licence Ouverte |
| **Unit of observation** | Program cell (year × institution × discipline) |
| **Observations (after filtering)** | ~10 584 |
| **Survey timings** | 18 months and 30 months after graduation |

The raw CSV uses **latin-1 encoding** and semicolon separators.

### Key variables

| Variable | Description |
|---|---|
| `taux_dinsertion` | **Target** — insertion rate (% employed among active graduates) |
| `domaine` / `discipline` | Broad field and specific discipline of study |
| `diplome` | Degree type |
| `academie` | Location of university (city) |
| `situation` | Survey timing (18 or 30 months after degree) |
| `emplois_cadre_ou_professions_intermediaires` | % in executive or intermediate-level positions |
| `emplois_stables` | % in stable employment contracts |
| `emplois_a_temps_plein` | % in full-time positions |
| `taux_de_chomage_regional` | Regional unemployment rate |
| `salaire_net_mensuel_median_regional` | Median regional monthly salary of young people |
| `femmes` | % women among graduates |
| `de_diplomes_boursiers` | % scholarship holders |

---

## Repository structure

```
ml_masters/
│
├── data/
│   └── raw/
│       └── fr-esr-insertion_professionnelle-master.csv
│
├── src/
│   ├── data_cleaning.py   # Loading, filtering, cleaning → exports filtered datasets
│   ├── features.py        # Feature engineering, encoding, scaling, target construction
│   ├── plots.py           # EDA visualisations (plots 1–5, 7)
│   ├── evaluate.py        # Additional diagnostics and EDA (correlation matrix, binary EDA)
│   └── train.py           # Model training: Linear/Ridge/Lasso regression + Logistic/RF classification
│
├── outputs/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## Pipeline

### 1. `src/data_cleaning.py`

- Loads CSV with `latin-1` encoding from `data/raw/`
- Keeps observations with **≥ 30 responses** and **≥ 50 % response rate**
- Replaces placeholder strings (`"nd"`, `"ns"`, `"nan"`, `"."`) with `NA`
- Converts numerical columns to `float64` (`errors="coerce"`)
- Drops columns with **> 70 % missing values** and highly correlated redundant columns
- Exports several filtered dataframes imported by downstream modules

### 2. `src/features.py`

- Defines categorical and numerical feature sets
- Constructs a proxy for program size (respondents ÷ response rate)
- Encodes the binary target: **Low Integration = 1** if insertion rate ≤ 85 (logistic models) or ≤ P25 ≈ 86 % (random forest)
- Builds `ColumnTransformer` preprocessors (OneHotEncoder + StandardScaler)

### 3. `src/plots.py`

| Output | Description |
|---|---|
| `plot_1.png` | Histogram of insertion rates with P25 threshold |
| `plot_2.png` | KDE — Low vs Normal Integration, by survey timing (18 / 30 months) |
| `plot_3.png` | Insertion rate boxplots by field |
| `plot_4.png` | Median salary boxplots by field |
| `plot_5.png` | 3-panel scatter: employment quality vs insertion rate, per-domain OLS lines |
| `plot_7.png` | Mean insertion rate over time by field |

### 4. `src/evaluate.py`

Supplementary EDA and diagnostics:
- Correlation matrix over numerical features
- Pairplot and binned scatter of response variables
- Distribution of categorical features
- Numerical and categorical feature distributions conditioned on Low Integration

### 5. `src/train.py`

Trains and evaluates all models via 5-fold cross-validation:

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Continuous | OLS baseline; MSE ≈ 25.4 on CV |
| Ridge Regression | Continuous | λ tuned over {0.01 … 10 000}; optimal λ = 100 |
| Lasso Regression | Continuous | λ tuned over {0.0001 … 100}; optimal λ = 0.001 |
| Logistic Regression | Binary | Unbalanced / class-weighted / threshold-adjusted variants |
| Random Forest | Binary | One-hot encoded features, sampling weights by discipline |

---

## Modelling approach

### Continuous prediction (Sections 3)

Linear, Ridge, and Lasso regressors are trained on year, diploma type, institution, location, department, discipline, months since graduation, % scholarship holders, % women, regional unemployment rate, and regional salary quartiles. A proxy for program size is also included.

Cross-validated MSE is ~25.4 across all three models. Test-set MSE is ~22.9. The low variation in insertion rates (IQR ≈ 9 pp, σ ≈ 6.7 pp) limits further gains.

### Binary classification (Sections 4–5)

Programs are labelled **low integration** if their insertion rate falls below the 25th percentile (~85–86 %). Models are evaluated on accuracy, sensitivity, specificity, F1-score, and ROC-AUC (≈ 0.85 for Logistic Regression).

Class imbalance (~25 % low integration) is handled via:
- Class reweighting (`class_weight="balanced"`)
- Decision threshold adjustment (evaluated at 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)
- Sampling weights in Random Forest

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

```bash
python src/data_cleaning.py   # verify filtering output
python src/plots.py           # generate EDA figures → outputs/figures/
python src/evaluate.py        # generate diagnostics figures → outputs/figures/
python src/train.py           # run all models
```

---

## References

1. Ministère de l'Enseignement supérieur et de la Recherche. *Insertion professionnelle des diplômés de Master.*
   https://data.enseignementsup-recherche.gouv.fr/explore/dataset/fr-esr-insertion_professionnelle-master/

2. data.gouv.fr. *Catalogue des datasets pour le Machine Learning.*
   https://www.data.gouv.fr/pages/donnees_apprentissage-automatique
