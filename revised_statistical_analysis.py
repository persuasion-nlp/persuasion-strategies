"""
Revised Statistical Analysis for LREC 2026 paper.
Addresses reviewer concerns:
  1. FDR (Benjamini-Hochberg) correction alongside Bonferroni
  2. Logistic regression (multivariate analysis)
  3. Proper reporting of which correction Reciprocity passes/fails
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. BUILD DIALOG-LEVEL DATASET
# ============================================================

print("=" * 70)
print("BUILDING DIALOG-LEVEL DATASET")
print("=" * 70)

dialog_data = pd.read_csv('full_dialog_with_all_analysis.csv')
donation_info = pd.read_csv('donation_dataset_stats.csv')
sentiment_data = pd.read_csv('sentiment_donation_by_dialog.csv')

# Persuader turns only
persuader = dialog_data[dialog_data['B4'] == 0].copy()
target = dialog_data[dialog_data['B4'] == 1].copy()

print(f"Total dialogs: {dialog_data['B2'].nunique()}")
print(f"Persuader turns: {len(persuader)}")
print(f"Target turns: {len(target)}")

# Strategy category mapping (from strategies_hierarchical.py)
STRATEGY_TO_CATEGORY = {
    "Rewarding Activity": "Exchange / Incentives",
    "Pre-giving": "Exchange / Incentives",
    "Reciprocity": "Exchange / Incentives",
    "Debt": "Exchange / Incentives",
    "Expertise": "Authority / Expertise",
    "Authority": "Authority / Expertise",
    "Credibility Appeal": "Authority / Expertise",
    "Moral Appeal": "Norms / Morality / Values",
    "Appeal to Values": "Norms / Morality / Values",
    "Activation of Impersonal Commitment": "Norms / Morality / Values",
    "Guilt Induction": "Norms / Morality / Values",
    "Self-feeling Appeal": "Norms / Morality / Values",
    "Activation of Personal Commitment": "Commitment / Consistency",
    "Commitment and Consistency": "Commitment / Consistency",
    "Foot-in-the-door": "Commitment / Consistency",
    "Door-in-the-face": "Commitment / Consistency",
    "Social Proof": "Social Influence",
    "Unity": "Social Influence",
    "Social Positioning": "Social Influence",
    "Rational Appeal": "Rational / Impact Appeal",
    "Logical Appeal": "Rational / Impact Appeal",
    "Emotional Appeal": "Emotional Influence",
    "Storytelling": "Emotional Influence",
    "Empathy Appeal": "Emotional Influence",
    "Sympathy Appeal": "Emotional Influence",
    "Fear Appeal": "Emotional Influence",
    "Emotional Manipulation": "Emotional Influence",
    "Urgency": "Urgency / Scarcity",
    "Scarcity": "Urgency / Scarcity",
    "Threat": "Threat / Pressure",
    "Aversive Stimulation": "Threat / Pressure",
    "Punishing Activity": "Threat / Pressure",
    "Overloading": "Threat / Pressure",
    "Confusion Induction": "Threat / Pressure",
    "Call to Action": "Call to Action",
    "Liking": "Call to Action",
    "Framing": "Framing & Presentation",
    "Loss Aversion Appeal": "Framing & Presentation",
    "Bait-and-switch": "Framing & Presentation",
    "Pretexting": "Framing & Presentation",
    # Conversation management (excluded from persuasion analysis)
    "Greeting / Rapport": "Conversation Management",
    "Permission / Time Check": "Conversation Management",
    "Charity Awareness Probe": "Conversation Management",
    "Qualification / Segmentation": "Conversation Management",
    "Donation Baseline / Habit Probe": "Conversation Management",
    "Logistics / Coordination": "Conversation Management",
    "Acknowledgement": "Conversation Management",
    "Conversation Closing": "Conversation Management",
    "Non-persuasive Other": "Conversation Management",
}

PERSUASION_CATEGORIES = [
    "Exchange / Incentives", "Authority / Expertise",
    "Norms / Morality / Values", "Commitment / Consistency",
    "Social Influence", "Rational / Impact Appeal",
    "Emotional Influence", "Urgency / Scarcity",
    "Threat / Pressure", "Call to Action", "Framing & Presentation",
]

# Individual strategies (excluding conversation management)
PERSUASION_STRATEGIES = [s for s, c in STRATEGY_TO_CATEGORY.items()
                         if c != "Conversation Management"]

persuader['category'] = persuader['strategy_ollama_single'].map(STRATEGY_TO_CATEGORY)

# Build dialog-level features: binary presence of each category and strategy
dialog_ids = dialog_data['B2'].unique()

# Category presence per dialog
cat_presence = {}
for dialog_id in dialog_ids:
    d_turns = persuader[persuader['B2'] == dialog_id]
    cats = set(d_turns['category'].dropna())
    cat_presence[dialog_id] = {cat: int(cat in cats) for cat in PERSUASION_CATEGORIES}

# Prefix category columns with "cat_" to avoid collision with strategy names
cat_df = pd.DataFrame.from_dict(cat_presence, orient='index')
cat_df.columns = ['cat_' + c for c in cat_df.columns]
cat_df.index.name = 'B2'
cat_df = cat_df.reset_index()

PERSUASION_CATEGORIES_COLS = ['cat_' + c for c in PERSUASION_CATEGORIES]

# Strategy presence per dialog
strat_presence = {}
for dialog_id in dialog_ids:
    d_turns = persuader[persuader['B2'] == dialog_id]
    strats = set(d_turns['strategy_ollama_single'].dropna())
    strat_presence[dialog_id] = {s: int(s in strats) for s in PERSUASION_STRATEGIES}

strat_df = pd.DataFrame.from_dict(strat_presence, orient='index')
strat_df.index.name = 'B2'
strat_df = strat_df.reset_index()

# Merge with donation info
# donation_dataset_stats counts if ANY participant (persuader or target) donated.
# The paper focuses on TARGET donations. sentiment_donation_by_dialog has the correct
# target-only 'donated' flag (545/1017 = 53.6%).
df = cat_df.merge(strat_df, on='B2', how='left')
df = df.merge(
    sentiment_data[['B2', 'donated', 'donation_amount', 'positive_rate', 'negative_rate', 'n_messages']],
    on='B2', how='left'
)

# Add interest features from target turns
# interest_label_ollama_v2: 0 = Not Interested, 1 = Neutral, 2 = Interested
# sentiment_ollama_v2: 'negative', 'neutral', 'positive'
target_features = target.copy()
target_features['interest_num'] = target_features['interest_label_ollama_v2'].astype(float)
target_features['sentiment_num'] = target_features['sentiment_ollama_v2'].map(
    {'negative': -1, 'neutral': 0, 'positive': 1}
).astype(float)

interest_by_dialog = target_features.groupby('B2').agg(
    mean_interest=('interest_num', 'mean'),
    mean_sentiment_num=('sentiment_num', 'mean'),
).reset_index()

df = df.merge(interest_by_dialog, on='B2', how='left')

df['donated'] = df['donated'].astype(int)
df['donation_amount'] = df['donation_amount'].fillna(0)

print(f"\nDialog-level dataset: {len(df)} dialogs")
print(f"Donation rate: {df['donated'].mean():.3f} ({df['donated'].sum()}/{len(df)})")
print(f"Mean donation: ${df['donation_amount'].mean():.2f}")

# ============================================================
# 2. CHI-SQUARE TESTS WITH BONFERRONI AND BH-FDR
# ============================================================

print("\n" + "=" * 70)
print("STRATEGY-LEVEL CHI-SQUARE TESTS: BONFERRONI vs BH-FDR")
print("=" * 70)

# --- 2a: CATEGORY-LEVEL (11 tests) ---
print("\n--- CATEGORY-LEVEL (11 comparisons) ---\n")

cat_results = []
for cat, col in zip(PERSUASION_CATEGORIES, PERSUASION_CATEGORIES_COLS):
    present = df[df[col] == 1]
    absent = df[df[col] == 0]
    don_present = present['donated'].sum()
    don_absent = absent['donated'].sum()
    n_present = len(present)
    n_absent = len(absent)

    if n_present < 5 or n_absent < 5:
        cat_results.append({
            'variable': cat, 'n_present': n_present, 'n_absent': n_absent,
            'rate_present': don_present / max(n_present, 1),
            'rate_absent': don_absent / max(n_absent, 1),
            'chi2': np.nan, 'p_raw': np.nan,
        })
        continue

    table = np.array([
        [don_present, n_present - don_present],
        [don_absent, n_absent - don_absent]
    ])
    chi2, p, _, _ = stats.chi2_contingency(table)

    cat_results.append({
        'variable': cat,
        'n_present': n_present,
        'n_absent': n_absent,
        'rate_present': don_present / n_present,
        'rate_absent': don_absent / n_absent,
        'chi2': chi2,
        'p_raw': p,
    })

cat_res_df = pd.DataFrame(cat_results)
valid = cat_res_df['p_raw'].notna()

# Bonferroni
cat_res_df.loc[valid, 'p_bonferroni'] = cat_res_df.loc[valid, 'p_raw'] * valid.sum()
cat_res_df['p_bonferroni'] = cat_res_df['p_bonferroni'].clip(upper=1.0)

# BH-FDR
if valid.sum() > 0:
    _, p_fdr, _, _ = multipletests(cat_res_df.loc[valid, 'p_raw'], method='fdr_bh')
    cat_res_df.loc[valid, 'p_fdr'] = p_fdr

cat_res_df = cat_res_df.sort_values('p_raw')

print(f"{'Category':<30} {'n':>5} {'Don%(+)':>8} {'Don%(-)':>8} {'chi2':>8} {'p_raw':>10} {'p_Bonf':>10} {'p_FDR':>10}")
print("-" * 100)
for _, r in cat_res_df.iterrows():
    sig_bonf = "***" if r.get('p_bonferroni', 1) < 0.001 else "**" if r.get('p_bonferroni', 1) < 0.01 else "*" if r.get('p_bonferroni', 1) < 0.05 else ""
    sig_fdr = "***" if r.get('p_fdr', 1) < 0.001 else "**" if r.get('p_fdr', 1) < 0.01 else "*" if r.get('p_fdr', 1) < 0.05 else ""
    print(f"{r['variable']:<30} {r['n_present']:>5} {r['rate_present']:>7.1%} {r['rate_absent']:>7.1%} "
          f"{r['chi2']:>8.2f} {r['p_raw']:>10.4f} {r.get('p_bonferroni', np.nan):>8.4f}{sig_bonf:>2} {r.get('p_fdr', np.nan):>8.4f}{sig_fdr:>2}")

# --- 2b: STRATEGY-LEVEL (individual strategies with n >= 20) ---
print("\n\n--- STRATEGY-LEVEL (individual strategies, n >= 20) ---\n")

strat_results = []
for strat in PERSUASION_STRATEGIES:
    present = df[df[strat] == 1]
    absent = df[df[strat] == 0]
    n_present = len(present)
    n_absent = len(absent)

    if n_present < 20:
        continue

    don_present = present['donated'].sum()
    don_absent = absent['donated'].sum()

    table = np.array([
        [don_present, n_present - don_present],
        [don_absent, n_absent - don_absent]
    ])
    chi2, p, _, _ = stats.chi2_contingency(table)

    # Effect size (difference in donation rates)
    rate_p = don_present / n_present
    rate_a = don_absent / n_absent
    delta = rate_p - rate_a

    strat_results.append({
        'strategy': strat,
        'category': STRATEGY_TO_CATEGORY[strat],
        'n_present': n_present,
        'n_absent': n_absent,
        'rate_present': rate_p,
        'rate_absent': rate_a,
        'delta': delta,
        'chi2': chi2,
        'p_raw': p,
    })

strat_res_df = pd.DataFrame(strat_results)
n_tests = len(strat_res_df)

# Bonferroni
strat_res_df['p_bonferroni'] = (strat_res_df['p_raw'] * n_tests).clip(upper=1.0)

# BH-FDR
_, p_fdr_strat, _, _ = multipletests(strat_res_df['p_raw'], method='fdr_bh')
strat_res_df['p_fdr'] = p_fdr_strat

strat_res_df = strat_res_df.sort_values('p_raw')

print(f"Number of tests: {n_tests}")
print(f"Bonferroni threshold: {0.05/n_tests:.5f}")
print()
print(f"{'Strategy':<35} {'n':>5} {'Don%(+)':>8} {'Don%(-)':>8} {'Delta':>7} {'chi2':>7} {'p_raw':>10} {'p_Bonf':>10} {'p_FDR':>10}")
print("-" * 115)
for _, r in strat_res_df.iterrows():
    sig_b = "***" if r['p_bonferroni'] < 0.001 else "**" if r['p_bonferroni'] < 0.01 else "*" if r['p_bonferroni'] < 0.05 else ""
    sig_f = "***" if r['p_fdr'] < 0.001 else "**" if r['p_fdr'] < 0.01 else "*" if r['p_fdr'] < 0.05 else ""
    print(f"{r['strategy']:<35} {r['n_present']:>5} {r['rate_present']:>7.1%} {r['rate_absent']:>7.1%} "
          f"{r['delta']:>+6.1%} {r['chi2']:>7.1f} {r['p_raw']:>10.6f} {r['p_bonferroni']:>8.4f}{sig_b:>2} {r['p_fdr']:>8.4f}{sig_f:>2}")

# Key finding: Reciprocity status
recip = strat_res_df[strat_res_df['strategy'] == 'Reciprocity']
if len(recip) > 0:
    r = recip.iloc[0]
    print(f"\n>>> RECIPROCITY STATUS:")
    print(f"    p_raw = {r['p_raw']:.6f}")
    print(f"    p_Bonferroni ({n_tests} tests) = {r['p_bonferroni']:.4f} {'SIGNIFICANT' if r['p_bonferroni'] < 0.05 else 'NOT SIGNIFICANT'}")
    print(f"    p_FDR (BH) = {r['p_fdr']:.4f} {'SIGNIFICANT' if r['p_fdr'] < 0.05 else 'NOT SIGNIFICANT'}")

guilt = strat_res_df[strat_res_df['strategy'] == 'Guilt Induction']
if len(guilt) > 0:
    r = guilt.iloc[0]
    print(f"\n>>> GUILT INDUCTION STATUS:")
    print(f"    p_raw = {r['p_raw']:.6f}")
    print(f"    p_Bonferroni ({n_tests} tests) = {r['p_bonferroni']:.6f} {'SIGNIFICANT' if r['p_bonferroni'] < 0.05 else 'NOT SIGNIFICANT'}")
    print(f"    p_FDR (BH) = {r['p_fdr']:.6f} {'SIGNIFICANT' if r['p_fdr'] < 0.05 else 'NOT SIGNIFICANT'}")

# ============================================================
# 3. LOGISTIC REGRESSION
# ============================================================

print("\n\n" + "=" * 70)
print("LOGISTIC REGRESSION: MULTIVARIATE ANALYSIS")
print("=" * 70)

import statsmodels.api as sm

# --- 3a: Model with category-level predictors ---
print("\n--- Model 1: Strategy CATEGORIES as predictors ---\n")

feature_cols_cat = PERSUASION_CATEGORIES_COLS.copy()
X_cat = df[feature_cols_cat].fillna(0).astype(float)
y = df['donated'].astype(float)

# Drop columns with near-zero variance (e.g., Threat/Pressure, Urgency/Scarcity)
low_var = X_cat.columns[X_cat.sum() < 20]
if len(low_var) > 0:
    print(f"Dropping low-variance categories (n < 20): {list(low_var)}")
    X_cat = X_cat.drop(columns=low_var)
    feature_cols_cat = [c for c in feature_cols_cat if c not in low_var]

X_cat_const = sm.add_constant(X_cat)

try:
    model_cat = sm.Logit(y, X_cat_const).fit(disp=0)
    print(model_cat.summary2())

    print("\n--- Odds Ratios (Category Model) ---")
    or_df = pd.DataFrame({
        'OR': np.exp(model_cat.params),
        'CI_low': np.exp(model_cat.conf_int()[0]),
        'CI_high': np.exp(model_cat.conf_int()[1]),
        'p': model_cat.pvalues,
    })
    or_df = or_df.drop('const', errors='ignore')
    or_df = or_df.sort_values('p')

    print(f"\n{'Predictor':<35} {'OR':>7} {'95% CI':>18} {'p':>10} {'Sig':>5}")
    print("-" * 80)
    for name, r in or_df.iterrows():
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
        print(f"{name:<35} {r['OR']:>7.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] {r['p']:>10.4f} {sig:>5}")

    print(f"\nModel fit: AIC = {model_cat.aic:.1f}, Pseudo R² = {model_cat.prsquared:.4f}")
    print(f"Log-likelihood: {model_cat.llf:.1f}")
    print(f"LLR p-value: {model_cat.llr_pvalue:.6f}")

except Exception as e:
    print(f"Category model failed: {e}")

# --- 3b: Model with categories + sentiment + interest ---
print("\n\n--- Model 2: Categories + Sentiment + Interest ---\n")

df['mean_sentiment'] = df['mean_sentiment_num'].fillna(0)
df['mean_interest_val'] = df['mean_interest'].fillna(1.0)  # neutral = 1
df['positive_rate_frac'] = df['positive_rate'].fillna(0) / 100.0
df['negative_rate_frac'] = df['negative_rate'].fillna(0) / 100.0

feature_cols_full = feature_cols_cat + ['mean_sentiment', 'mean_interest_val']
X_full = df[feature_cols_full].fillna(0).astype(float)
X_full_const = sm.add_constant(X_full)

try:
    model_full = sm.Logit(y, X_full_const).fit(disp=0)
    print(model_full.summary2())

    print("\n--- Odds Ratios (Full Model) ---")
    or_full = pd.DataFrame({
        'OR': np.exp(model_full.params),
        'CI_low': np.exp(model_full.conf_int()[0]),
        'CI_high': np.exp(model_full.conf_int()[1]),
        'p': model_full.pvalues,
    })
    or_full = or_full.drop('const', errors='ignore')
    or_full = or_full.sort_values('p')

    print(f"\n{'Predictor':<35} {'OR':>7} {'95% CI':>18} {'p':>10} {'Sig':>5}")
    print("-" * 80)
    for name, r in or_full.iterrows():
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
        print(f"{name:<35} {r['OR']:>7.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] {r['p']:>10.4f} {sig:>5}")

    print(f"\nModel fit: AIC = {model_full.aic:.1f}, Pseudo R² = {model_full.prsquared:.4f}")
    print(f"Log-likelihood: {model_full.llf:.1f}")
    print(f"LLR p-value: {model_full.llr_pvalue:.6f}")

    # Likelihood ratio test: full vs categories-only
    lr_stat = 2 * (model_full.llf - model_cat.llf)
    lr_df = len(feature_cols_full) - len(feature_cols_cat)
    lr_p = stats.chi2.sf(lr_stat, lr_df)
    print(f"\nLR test (full vs categories-only): chi2 = {lr_stat:.2f}, df = {lr_df}, p = {lr_p:.6f}")
    if lr_p < 0.05:
        print(">>> Sentiment/Interest significantly improve the model over categories alone.")

except Exception as e:
    print(f"Full model failed: {e}")

# --- 3c: Model with individual key strategies + sentiment + interest ---
print("\n\n--- Model 3: Key Strategies (Guilt + Reciprocity) + Sentiment + Interest ---\n")

feature_cols_key = ['Guilt Induction', 'Reciprocity', 'mean_sentiment', 'mean_interest_val']
X_key = df[feature_cols_key].fillna(0).astype(float)
X_key_const = sm.add_constant(X_key)

try:
    model_key = sm.Logit(y, X_key_const).fit(disp=0)
    print(model_key.summary2())

    print("\n--- Odds Ratios (Key Strategies Model) ---")
    or_key = pd.DataFrame({
        'OR': np.exp(model_key.params),
        'CI_low': np.exp(model_key.conf_int()[0]),
        'CI_high': np.exp(model_key.conf_int()[1]),
        'p': model_key.pvalues,
    })
    or_key = or_key.drop('const', errors='ignore')
    or_key = or_key.sort_values('p')

    print(f"\n{'Predictor':<35} {'OR':>7} {'95% CI':>18} {'p':>10} {'Sig':>5}")
    print("-" * 80)
    for name, r in or_key.iterrows():
        sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
        print(f"{name:<35} {r['OR']:>7.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}] {r['p']:>10.4f} {sig:>5}")

    print(f"\nModel fit: AIC = {model_key.aic:.1f}, Pseudo R² = {model_key.prsquared:.4f}")

except Exception as e:
    print(f"Key model failed: {e}")

# ============================================================
# 4. SUMMARY FOR PAPER
# ============================================================

print("\n\n" + "=" * 70)
print("SUMMARY FOR PAPER REVISION")
print("=" * 70)

print("""
KEY CORRECTIONS:

1. RECIPROCITY:
   - The paper claims Reciprocity is significant after Bonferroni (p=0.034)
   - This needs to be checked: with N strategy-level tests, Bonferroni
     threshold = 0.05/N

2. FDR (Benjamini-Hochberg) is LESS conservative than Bonferroni
   and is standard for exploratory analyses with many comparisons.
   Adding both corrections strengthens the methodology.

3. LOGISTIC REGRESSION demonstrates whether:
   - Strategy categories predict donation AFTER controlling for other categories
   - Sentiment/interest add explanatory power beyond strategies
   - Guilt Induction remains significant in multivariate context

This addresses the reviewer's concern about confounding and the absence
of multivariate analysis.
""")

# Save results
cat_res_df.to_csv('revised_category_tests.csv', index=False)
strat_res_df.to_csv('revised_strategy_tests.csv', index=False)
print("Results saved to revised_category_tests.csv and revised_strategy_tests.csv")
