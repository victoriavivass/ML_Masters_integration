import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

_FIGURES = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Import cleaned DataFrame from data_cleaning.py
# ---------------------------------------------------------------------------
from data_cleaning import masters_data_filtered

# ---------------------------------------------------------------------------
# Style & palette
# ---------------------------------------------------------------------------
plt.style.use("default")

RED   = "#EF4135"
BLUE  = "#002395"
WHITE = "#FFFFFF"
BLACK = "#1a1a1a"

DOMAIN_PALETTE = {
    "DEG": RED,
    "SHS": BLUE,
    "STS": "#1a5c38",
    "LLA": "#888888",
}
DOMAIN_LABELS = {
    "DEG": "Law & Economics",
    "SHS": "Humanities & Social Sci.",
    "STS": "Science & Technology",
    "LLA": "Languages & Arts",
}

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "fontfamily": "serif", "color": "black"}
FONT_LABEL  = {"fontsize": 11, "fontfamily": "sans-serif", "color": "black"}
GRID_KWARGS = {"color": "#cccccc", "linewidth": 0.5, "alpha": 0.8}
LEGEND_KW   = {"fontsize": 10, "labelcolor": "black", "framealpha": 0.0, "edgecolor": "none"}

def _style_ax(ax):
    """Apply black text / white background to any axes."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(colors="black", which="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

def _base_fig(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    _style_ax(ax)
    return fig, ax

def _save(fig, name):
    path = os.path.join(_FIGURES, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
df = masters_data_filtered.copy()
threshold        = df["taux_dinsertion"].quantile(0.25)   # P25 – used in plot 1
median_threshold = df["taux_dinsertion"].median()          # median – used in plot 2

# ===========================================================================
# Plot 1 – Histogram of taux_dinsertion with p25 line
# ===========================================================================
fig, ax = _base_fig()

ax.hist(
    df["taux_dinsertion"].dropna(),
    bins=40,
    color=BLUE,
    edgecolor=WHITE,
    linewidth=0.4,
    alpha=0.85,
)
ax.axvline(threshold, color=RED, linewidth=1.8, linestyle="--",
           label=f"P25 = {threshold:.1f}%")

ax.set_title("Distribution of Insertion Rates", **FONT_TITLE)
ax.set_xlabel("Insertion Rate (%)", **FONT_LABEL)
ax.set_ylabel("Count", **FONT_LABEL)
ax.legend(**LEGEND_KW)

_save(fig, "plot_1.png")

# ===========================================================================
# Plot 2 – KDE distributions: Low vs Normal Integration, 2 panels by situation
# ===========================================================================
# Derive situation keys directly from the data to avoid encoding issues
_sit_vals = sorted(df["situation"].dropna().unique(), key=lambda s: int("".join(filter(str.isdigit, s)) or 0))
SITUATION_PANELS = [
    (_sit_vals[0], "18 Months After Graduation"),
    (_sit_vals[1], "30 Months After Graduation"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.patch.set_facecolor("white")

for ax, (sit_val, sit_label) in zip(axes, SITUATION_PANELS):
    _style_ax(ax)

    sit_df = df[df["situation"] == sit_val]["taux_dinsertion"].dropna()

    if len(sit_df) == 0:
        ax.set_title(f"{sit_label}\n(no data)", **FONT_TITLE)
        continue

    sit_threshold = sit_df.quantile(0.25)
    low_vals    = sit_df[sit_df <  sit_threshold]
    normal_vals = sit_df[sit_df >= sit_threshold]

    x_grid = np.linspace(sit_df.min() - 5, sit_df.max() + 5, 500)

    if len(low_vals) > 1:
        kde_low = gaussian_kde(low_vals, bw_method="scott")
        ax.plot(x_grid, kde_low(x_grid), color=RED, linewidth=2, label="Low Integration")
        ax.fill_between(x_grid, kde_low(x_grid), color=RED, alpha=0.25)

    if len(normal_vals) > 1:
        kde_normal = gaussian_kde(normal_vals, bw_method="scott")
        ax.plot(x_grid, kde_normal(x_grid), color=BLUE, linewidth=2, label="Normal Integration")
        ax.fill_between(x_grid, kde_normal(x_grid), color=BLUE, alpha=0.25)

    ax.axvline(sit_threshold, color="#555555", linewidth=1.2, linestyle="--",
               label=f"P25 = {sit_threshold:.1f}%")

    ax.set_title(sit_label, **FONT_TITLE)
    ax.set_xlabel("Insertion Rate (%)", **FONT_LABEL)
    ax.legend(**LEGEND_KW)

axes[0].set_ylabel("Density", **FONT_LABEL)
fig.suptitle("Class Balance by Situation", **FONT_TITLE, y=1.02)
plt.tight_layout()

_save(fig, "plot_2.png")

# ===========================================================================
# Plot 3 – Boxplots of taux_dinsertion by domaine (ordered by median)
# ===========================================================================
domain_col = "code_du_domaine"
domains_present = [d for d in ["DEG", "SHS", "STS", "LLA"]
                   if d in df[domain_col].unique()]

domain_medians = (
    df[df[domain_col].isin(domains_present)]
    .groupby(domain_col)["taux_dinsertion"]
    .median()
    .sort_values()
)
ordered_domains = domain_medians.index.tolist()

data_by_domain = [
    df.loc[df[domain_col] == d, "taux_dinsertion"].dropna().values
    for d in ordered_domains
]
labels_by_domain = [DOMAIN_LABELS.get(d, d) for d in ordered_domains]
colors_by_domain = [DOMAIN_PALETTE.get(d, "#888888") for d in ordered_domains]

fig, ax = _base_fig(figsize=(10, 6))
bp = ax.boxplot(
    data_by_domain,
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 2},
    whiskerprops={"color": "#555555"},
    capprops={"color": "#555555"},
    flierprops={"marker": "o", "markerfacecolor": "#888888", "markersize": 2, "alpha": 0.4},
)
for patch, color in zip(bp["boxes"], colors_by_domain):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks(range(1, len(ordered_domains) + 1))
ax.set_xticklabels(labels_by_domain, fontsize=10)
ax.set_title("Insertion Rate by Field", **FONT_TITLE)
ax.set_ylabel("Insertion Rate (%)", **FONT_LABEL)

ax.set_ylim(50, 100)

_save(fig, "plot_3.png")

# ===========================================================================
# Plot 4 – Boxplots of median salary by domaine (same style as plot 3)
# ===========================================================================
salary_col = "salaire_net_median_des_emplois_a_temps_plein"

salary_medians = (
    df[df[domain_col].isin(domains_present)]
    .groupby(domain_col)[salary_col]
    .median()
    .sort_values()
)
ordered_domains_sal = salary_medians.index.tolist()

data_by_domain_sal = [
    df.loc[df[domain_col] == d, salary_col].dropna().values
    for d in ordered_domains_sal
]
labels_by_domain_sal = [DOMAIN_LABELS.get(d, d) for d in ordered_domains_sal]
colors_by_domain_sal = [DOMAIN_PALETTE.get(d, "#888888") for d in ordered_domains_sal]

fig, ax = _base_fig(figsize=(10, 6))
bp = ax.boxplot(
    data_by_domain_sal,
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 2},
    whiskerprops={"color": "#555555"},
    capprops={"color": "#555555"},
    flierprops={"marker": "o", "markerfacecolor": "#888888", "markersize": 2, "alpha": 0.4},
)
for patch, color in zip(bp["boxes"], colors_by_domain_sal):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks(range(1, len(ordered_domains_sal) + 1))
ax.set_xticklabels(labels_by_domain_sal, fontsize=10)
ax.set_title("Median Net Salary by Field", **FONT_TITLE)
ax.set_ylabel("Median Net Salary (€/month)", **FONT_LABEL)

_save(fig, "plot_4.png")

# ===========================================================================
# Plot 5 – 3-panel scatter with regression lines: employment quality vs insertion rate
# ===========================================================================
from scipy.stats import linregress

PANELS = [
    ("emplois_cadre_ou_professions_intermediaires", "Upper/Intermediate Jobs (%)"),
    ("emplois_stables",                             "Stable Employment (%)"),
    ("emplois_a_temps_plein",                       "Full-time Employment (%)"),
]

p25_insertion = df["taux_dinsertion"].quantile(0.25)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.patch.set_facecolor("white")

for ax, (x_col, x_label) in zip(axes, PANELS):
    _style_ax(ax)

    panel_df = df[[x_col, "taux_dinsertion", domain_col]].dropna()

    for domain, group in panel_df.groupby(domain_col):
        color = DOMAIN_PALETTE.get(domain, "#888888")
        label = DOMAIN_LABELS.get(domain, domain)

        ax.scatter(
            group[x_col],
            group["taux_dinsertion"],
            color=color,
            alpha=0.15,
            s=15,
            edgecolors="none",
            label=label,
        )

        # Regression line
        slope, intercept, *_ = linregress(group[x_col], group["taux_dinsertion"])
        x_range = np.linspace(group[x_col].min(), group[x_col].max(), 200)
        ax.plot(x_range, intercept + slope * x_range,
                color=color, linewidth=1.5)

    # P25 threshold horizontal line (label added to shared legend below)
    ax.axhline(p25_insertion, color="#888888", linewidth=1.2, linestyle="--")

    ax.set_xlabel(x_label, **FONT_LABEL)
    ax.set_title(x_label, **FONT_TITLE)

axes[0].set_ylabel("Insertion Rate (%)", **FONT_LABEL)

# Shared legend (domains + threshold entry)
handles = [
    mpatches.Patch(color=DOMAIN_PALETTE.get(d, "#888888"), label=DOMAIN_LABELS.get(d, d))
    for d in domains_present
]
handles.append(
    plt.Line2D([0], [0], color="#888888", linewidth=1.2, linestyle="--",
               label=f"Low integration threshold (P25 = {p25_insertion:.1f}%)")
)
fig.legend(handles=handles, loc="lower center", ncol=len(handles),
           fontsize=10, labelcolor="black", framealpha=0.0, edgecolor="none",
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle("Employment Quality vs Insertion Rate by Field", **FONT_TITLE, y=1.02)
plt.tight_layout()

_save(fig, "plot_5.png")

# ===========================================================================
# Plot 7 – Temporal trend by domaine
# ===========================================================================
annee_col = "annee"
yearly_domain = (
    df[df[domain_col].isin(domains_present)]
    .groupby([annee_col, domain_col])["taux_dinsertion"]
    .mean()
    .reset_index()
    .sort_values(annee_col)
)

fig, ax = _base_fig(figsize=(11, 6))

for domain in domains_present:
    subset = yearly_domain[yearly_domain[domain_col] == domain]
    ax.plot(
        subset[annee_col],
        subset["taux_dinsertion"],
        color=DOMAIN_PALETTE.get(domain, "#888888"),
        linewidth=2,
        marker="o",
        markersize=4,
        label=DOMAIN_LABELS.get(domain, domain),
    )

ax.set_title("Insertion Rate Over Time by Field", **FONT_TITLE)
ax.set_xlabel("Year", **FONT_LABEL)
ax.set_ylabel("Mean Insertion Rate (%)", **FONT_LABEL)
ax.legend(**{**LEGEND_KW, "fontsize": 9})
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

_save(fig, "plot_7.png")
