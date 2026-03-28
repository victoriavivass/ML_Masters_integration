# Predicting Professional Integration After French Master's Programs

**Authors:** Célina Madaschi · Ana Catarina da Fonseca Gaspar · María Victoria Vivas Gutiérrez

---

## Objective

Binary classification task: predict whether a Master's program has a **low professional integration rate** 30 months after graduation. The goal is to identify vulnerable programs and guide student information or institutional support.

The classification threshold is set at the **25th percentile** of the insertion rate distribution.

---

## Dataset

| Field | Value |
|---|---|
| **Name** | Insertion professionnelle des diplômés de Master en universités et établissements assimilés |
| **Source** | Ministère de l'Enseignement supérieur et de la Recherche (MESR-SIES) |
| **Licence** | Open Data — Licence Ouverte |
| **Unit of observation** | Program cell (year × institution × discipline) |
| **Observations (filtered)** | ~10 584 |
| **Survey timings** | 18 months and 30 months after graduation |

The raw CSV uses **latin-1 encoding** and semicolon separators.

### Key variables

| Variable | Description |
|---|---|
| `taux_dinsertion` | **Target** — insertion rate (employment among active graduates) |
| `code_du_domaine` | Broad field: DEG (Law & Econ), SHS (Humanities), STS (Science & Tech), LLA (Languages) |
| `emplois_cadre_ou_professions_intermediaires` | % in executive or intermediate-level positions |
| `emplois_stables` | % in stable employment contracts |
| `emplois_a_temps_plein` | % in full-time positions |
| `salaire_net_median_des_emplois_a_temps_plein` | Median net monthly salary (full-time) |
| `taux_de_chomage_regional` | Regional unemployment rate |
| `femmes` | % women among graduates |
| `de_diplomes_boursiers` | % scholarship holders |
| `situation` | Survey timing (18 or 30 months after degree) |

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
│   ├── data_cleaning.py      # Loading, filtering, cleaning → exports masters_data_filtered
│   ├── plots.py              # EDA visualisations
│   ├── features.py           # (planned) feature engineering
│   ├── train.py              # (planned) model training
│   └── evaluate.py           # (planned) evaluation & metrics
│
├── outputs/
│   └── figures/
│       ├── plot_1.png        # Insertion rate distribution + P25
│       ├── plot_2.png        # Class balance KDE by survey timing
│       ├── plot_3.png        # Insertion rate by field (boxplot)
│       ├── plot_4.png        # Median salary by field (boxplot)
│       ├── plot_5.png        # Employment quality vs insertion (scatter + regression)
│       └── plot_7.png        # Insertion rate over time by field
│
├── reports/
│   └── proposal.tex          # Project proposal (Overleaf-ready LaTeX)
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
- Drops columns with **> 70 % missing values**
- Exports `masters_data_filtered` — imported by all other modules

### 2. `src/plots.py`

| Output | Description |
|---|---|
| `plot_1.png` | Histogram of insertion rates with P25 threshold |
| `plot_2.png` | KDE — Low vs Normal Integration, by survey timing (18 / 30 months) |
| `plot_3.png` | Insertion rate boxplots by field |
| `plot_4.png` | Median salary boxplots by field |
| `plot_5.png` | 3-panel scatter: employment quality vs insertion rate, per-domain regression lines |
| `plot_7.png` | Mean insertion rate over time by field |

### 3. Planned modules

| Module | Role |
|---|---|
| `src/features.py` | Feature engineering, encoding, scaling |
| `src/train.py` | Model training (Logistic Regression, Lasso/ElasticNet, Random Forest / GBM) |
| `src/evaluate.py` | Cross-validation, ROC-AUC, F1, confusion matrix |

---

## Statistical approach

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Lasso / Elastic Net | Regularisation + automatic feature selection |
| Random Forest / Gradient Boosting | Non-linear relationships and interactions |

Evaluation: **ROC-AUC**, **F1-score**, precision/recall, confusion matrix — via cross-validation. Class imbalance explicitly monitored.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

```bash
python src/data_cleaning.py  # verify filtering output
python src/plots.py          # generate all figures → outputs/figures/
```

---

## References

1. Ministère de l'Enseignement supérieur et de la Recherche. *Insertion professionnelle des diplômés de Master.*
   https://data.enseignementsup-recherche.gouv.fr/explore/dataset/fr-esr-insertion_professionnelle-master/

2. data.gouv.fr. *Catalogue des datasets pour le Machine Learning.*
   https://www.data.gouv.fr/pages/donnees_apprentissage-automatique
