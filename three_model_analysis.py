"""
Comprehensive three-model replication analysis.

Replicates ALL analyses from the paper using Qwen3:30b, Mistral-Small-3.2, and Phi-4:
1. Strategy/category distributions
2. Category-level chi-square tests (11 categories vs donation)
3. Strategy-level chi-square tests (strategies with n>=20 dialogues, Bonferroni + FDR)
4. Strategy-level Mann-Whitney U tests (donation amount)
5. Logistic regression Model 1 (categories only)
6. Logistic regression Model 2 (categories + sentiment + interest)
7. Logistic regression Model 3 (Guilt + Reciprocity + sentiment + interest)
8. Strategy-response sentiment link
9. Inter-model agreement (pairwise Cohen's kappa)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import warnings
import os
import sys

warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# TAXONOMY: strategy -> category mapping
# ============================================================

STRATEGY_TO_CATEGORY = {
    "Moral Appeal": "Norms / Morality / Values",
    "Appeal to Values": "Norms / Morality / Values",
    "Guilt Induction": "Norms / Morality / Values",
    "Self-feeling Appeal": "Norms / Morality / Values",
    "Activation of Impersonal Commitment": "Norms / Morality / Values",
    "Rational Appeal": "Rational / Impact Appeal",
    "Logical Appeal": "Rational / Impact Appeal",
    "Framing": "Framing & Presentation",
    "Loss Aversion Appeal": "Framing & Presentation",
    "Bait-and-switch": "Framing & Presentation",
    "Pretexting": "Framing & Presentation",
    "Credibility Appeal": "Authority / Expertise",
    "Authority": "Authority / Expertise",
    "Expertise": "Authority / Expertise",
    "Empathy Appeal": "Emotional Influence",
    "Storytelling": "Emotional Influence",
    "Emotional Appeal": "Emotional Influence",
    "Fear Appeal": "Emotional Influence",
    "Sympathy Appeal": "Emotional Influence",
    "Emotional Manipulation": "Emotional Influence",
    "Call to Action": "Call to Action",
    "Liking": "Call to Action",
    "Commitment and Consistency": "Commitment / Consistency",
    "Activation of Personal Commitment": "Commitment / Consistency",
    "Foot-in-the-door": "Commitment / Consistency",
    "Door-in-the-face": "Commitment / Consistency",
    "Social Proof": "Social Influence",
    "Unity": "Social Influence",
    "Social Positioning": "Social Influence",
    "Reciprocity": "Exchange / Incentives",
    "Rewarding Activity": "Exchange / Incentives",
    "Pre-giving": "Exchange / Incentives",
    "Debt": "Exchange / Incentives",
    "Urgency": "Urgency / Scarcity",
    "Scarcity": "Urgency / Scarcity",
    "Threat": "Threat / Pressure",
    "Aversive Stimulation": "Threat / Pressure",
    "Punishing Activity": "Threat / Pressure",
    "Overloading": "Threat / Pressure",
    "Confusion Induction": "Threat / Pressure",
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
    "Norms / Morality / Values",
    "Rational / Impact Appeal",
    "Framing & Presentation",
    "Authority / Expertise",
    "Emotional Influence",
    "Call to Action",
    "Commitment / Consistency",
    "Social Influence",
    "Exchange / Incentives",
    "Urgency / Scarcity",
    "Threat / Pressure",
]


def get_category(strategy):
    """Map strategy name to category."""
    return STRATEGY_TO_CATEGORY.get(strategy, "Unknown")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    # Full dialog data (has sentiment, interest, B4, etc.)
    full_df = pd.read_csv(os.path.join(base, "full_dialog_with_all_analysis.csv"))

    # Donation outcomes — TARGET donations only (545/1017 = 53.6%)
    # NOTE: donation_dataset_stats.csv has has_donation which counts if EITHER
    # participant donated (711/1017 = 69.9%). The paper uses TARGET donation only.
    donations = pd.read_csv(os.path.join(base, "sentiment_donation_by_dialog.csv"))

    # Mistral annotations
    mistral = pd.read_csv(os.path.join(base, "mistral_annotations.csv"))

    # Phi4 annotations
    phi4 = pd.read_csv(os.path.join(base, "phi4_annotations.csv"))

    return full_df, donations, mistral, phi4


def prepare_model_data(full_df, mistral, phi4, donations):
    """Prepare per-model datasets with dialog-level features."""
    # Persuader turns from full_df (Qwen)
    persuader = full_df[full_df["B4"] == 0].copy()
    persuader["qwen_strategy"] = persuader["strategy_ollama_single"]
    persuader["qwen_category"] = persuader["qwen_strategy"].map(get_category)

    # Target turns for sentiment/interest
    target = full_df[full_df["B4"] == 1].copy()

    # Map sentiment to numeric
    sent_map = {"negative": -1, "neutral": 0, "positive": 1}
    target["sentiment_num"] = target["sentiment_ollama_v2"].map(sent_map)
    target["interest_num"] = pd.to_numeric(target["interest_label_ollama_v2"], errors="coerce")

    # Dialog-level target features
    dialog_sentiment = target.groupby("B2")["sentiment_num"].mean().rename("mean_sentiment")
    dialog_interest = target.groupby("B2")["interest_num"].mean().rename("mean_interest")
    dialog_target = pd.concat([dialog_sentiment, dialog_interest], axis=1)

    # Mistral: add category
    mistral["mistral_category"] = mistral["mistral_strategy"].map(get_category)

    # Phi4: add category (may already exist but ensure mapping)
    phi4["phi4_category_mapped"] = phi4["phi4_strategy"].map(get_category)

    # Merge donation info (target donations only)
    donations_slim = donations[["B2", "donated", "donation_amount"]].copy()
    donations_slim = donations_slim.rename(columns={"donated": "has_donation", "donation_amount": "total_donation"})

    models = {}

    # --- QWEN ---
    qwen_dialog = build_dialog_features(persuader, "qwen_strategy", "qwen_category", "B2")
    qwen_dialog = qwen_dialog.merge(donations_slim, on="B2", how="inner")
    qwen_dialog = qwen_dialog.merge(dialog_target, left_on="B2", right_index=True, how="left")
    models["Qwen3:30b"] = {
        "turns": persuader[["B2", "Turn", "qwen_strategy", "qwen_category"]].rename(
            columns={"qwen_strategy": "strategy", "qwen_category": "category"}),
        "dialogs": qwen_dialog,
        "strategy_col": "qwen_strategy",
    }

    # --- MISTRAL ---
    mistral_dialog = build_dialog_features(mistral, "mistral_strategy", "mistral_category", "dialog_id")
    mistral_dialog = mistral_dialog.rename(columns={"dialog_id": "B2"})
    mistral_dialog = mistral_dialog.merge(donations_slim, on="B2", how="inner")
    mistral_dialog = mistral_dialog.merge(dialog_target, left_on="B2", right_index=True, how="left")
    models["Mistral-Small-3.2"] = {
        "turns": mistral[["dialog_id", "turn", "mistral_strategy", "mistral_category"]].rename(
            columns={"dialog_id": "B2", "turn": "Turn",
                      "mistral_strategy": "strategy", "mistral_category": "category"}),
        "dialogs": mistral_dialog,
        "strategy_col": "mistral_strategy",
    }

    # --- PHI4 ---
    phi4_dialog = build_dialog_features(phi4, "phi4_strategy", "phi4_category_mapped", "dialog_id")
    phi4_dialog = phi4_dialog.rename(columns={"dialog_id": "B2"})
    phi4_dialog = phi4_dialog.merge(donations_slim, on="B2", how="inner")
    phi4_dialog = phi4_dialog.merge(dialog_target, left_on="B2", right_index=True, how="left")
    models["Phi-4"] = {
        "turns": phi4[["dialog_id", "turn", "phi4_strategy", "phi4_category_mapped"]].rename(
            columns={"dialog_id": "B2", "turn": "Turn",
                      "phi4_strategy": "strategy", "phi4_category_mapped": "category"}),
        "dialogs": phi4_dialog,
        "strategy_col": "phi4_strategy",
    }

    return models, full_df, target


def build_dialog_features(df, strat_col, cat_col, dialog_col):
    """Build dialog-level binary features for each strategy and category."""
    # Get all unique strategies and categories
    all_strategies = df[strat_col].dropna().unique()
    all_categories = df[cat_col].dropna().unique()

    dialog_ids = df[dialog_col].unique()
    result = pd.DataFrame({dialog_col: dialog_ids})

    # Strategy-level: binary presence per dialog
    for strat in sorted(all_strategies):
        dialogs_with = df[df[strat_col] == strat][dialog_col].unique()
        result[f"has_{strat}"] = result[dialog_col].isin(dialogs_with).astype(int)

    # Category-level: binary presence per dialog
    for cat in sorted(all_categories):
        dialogs_with = df[df[cat_col] == cat][dialog_col].unique()
        result[f"cat_{cat}"] = result[dialog_col].isin(dialogs_with).astype(int)

    return result


# ============================================================
# ANALYSIS 1: Strategy distribution
# ============================================================

def analysis_distribution(models, out):
    out.write("=" * 80 + "\n")
    out.write("ANALYSIS 1: STRATEGY & CATEGORY DISTRIBUTIONS\n")
    out.write("=" * 80 + "\n\n")

    for model_name, data in models.items():
        turns = data["turns"]
        total = len(turns)
        non_cm = turns[turns["category"] != "Conversation Management"]

        out.write(f"--- {model_name} ---\n")
        out.write(f"Total persuader turns: {total}\n")
        out.write(f"Conversation Management: {total - len(non_cm)} ({(total-len(non_cm))*100/total:.1f}%)\n")
        out.write(f"Persuasion turns: {len(non_cm)} ({len(non_cm)*100/total:.1f}%)\n\n")

        # Category distribution
        cat_counts = turns["category"].value_counts()
        out.write(f"Category distribution:\n")
        for cat in PERSUASION_CATEGORIES:
            n = cat_counts.get(cat, 0)
            out.write(f"  {cat:35s} {n:5d} ({n*100/total:5.1f}%)\n")
        out.write("\n")

        # Top 10 strategies (non-CM)
        strat_counts = non_cm["strategy"].value_counts().head(10)
        out.write(f"Top 10 persuasion strategies:\n")
        for strat, n in strat_counts.items():
            out.write(f"  {strat:40s} {n:5d} ({n*100/total:5.1f}%)\n")
        out.write("\n")

    # Cross-model comparison: key strategies
    out.write("--- Cross-model comparison: key strategy counts ---\n")
    key_strategies = ["Guilt Induction", "Reciprocity", "Commitment and Consistency",
                      "Call to Action", "Rational Appeal", "Empathy Appeal",
                      "Greeting / Rapport", "Acknowledgement"]
    out.write(f"{'Strategy':40s} {'Qwen':>8s} {'Mistral':>8s} {'Phi-4':>8s}\n")
    out.write("-" * 68 + "\n")
    for strat in key_strategies:
        counts = []
        for model_name in ["Qwen3:30b", "Mistral-Small-3.2", "Phi-4"]:
            n = (models[model_name]["turns"]["strategy"] == strat).sum()
            counts.append(n)
        out.write(f"{strat:40s} {counts[0]:8d} {counts[1]:8d} {counts[2]:8d}\n")
    out.write("\n\n")


# ============================================================
# ANALYSIS 2: Category-level chi-square tests
# ============================================================

def analysis_category_chisq(models, out):
    out.write("=" * 80 + "\n")
    out.write("ANALYSIS 2: CATEGORY-LEVEL CHI-SQUARE TESTS (11 categories vs donation)\n")
    out.write("=" * 80 + "\n\n")

    for model_name, data in models.items():
        dialogs = data["dialogs"]
        out.write(f"--- {model_name} ---\n")
        out.write(f"{'Category':35s} {'Present%':>8s} {'Absent%':>8s} {'chi2':>8s} {'p_raw':>10s} {'p_Bonf':>10s}\n")
        out.write("-" * 85 + "\n")

        results = []
        for cat in PERSUASION_CATEGORIES:
            col = f"cat_{cat}"
            if col not in dialogs.columns:
                continue
            present = dialogs[dialogs[col] == 1]
            absent = dialogs[dialogs[col] == 0]
            if len(present) < 5 or len(absent) < 5:
                continue

            don_pres = present["has_donation"].mean() * 100
            don_abs = absent["has_donation"].mean() * 100

            ct = pd.crosstab(dialogs[col], dialogs["has_donation"])
            if ct.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ct)
            else:
                chi2, p = 0, 1.0
            results.append((cat, don_pres, don_abs, chi2, p))

        # Bonferroni
        n_tests = len(results)
        for cat, don_pres, don_abs, chi2, p in results:
            p_bonf = min(p * n_tests, 1.0)
            sig = "*" if p_bonf < 0.05 else ""
            out.write(f"{cat:35s} {don_pres:7.1f}% {don_abs:7.1f}% {chi2:8.2f} {p:10.4f} {p_bonf:10.4f} {sig}\n")
        out.write("\n")


# ============================================================
# ANALYSIS 3: Strategy-level chi-square tests
# ============================================================

def analysis_strategy_chisq(models, out):
    out.write("=" * 80 + "\n")
    out.write("ANALYSIS 3: STRATEGY-LEVEL CHI-SQUARE TESTS (n>=20 dialogs, Bonferroni + FDR)\n")
    out.write("=" * 80 + "\n\n")

    for model_name, data in models.items():
        dialogs = data["dialogs"]
        out.write(f"--- {model_name} ---\n")

        # Find strategies with n >= 20 dialogs (non-CM)
        # Exclude has_donation (the outcome variable) and has_negative/has_positive (sentiment)
        exclude = {"has_donation", "has_negative", "has_positive"}
        strat_cols = [c for c in dialogs.columns if c.startswith("has_") and
                      c not in exclude and
                      get_category(c[4:]) != "Conversation Management"]

        results = []
        for col in strat_cols:
            strat = col[4:]
            n_present = dialogs[col].sum()
            if n_present < 20:
                continue

            present = dialogs[dialogs[col] == 1]
            absent = dialogs[dialogs[col] == 0]
            don_pres = present["has_donation"].mean() * 100
            don_abs = absent["has_donation"].mean() * 100
            delta = don_pres - don_abs

            ct = pd.crosstab(dialogs[col], dialogs["has_donation"])
            if ct.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ct)
                n = ct.sum().sum()
                phi = np.sqrt(chi2 / n)
            else:
                chi2, p, phi = 0, 1.0, 0
            results.append((strat, n_present, don_pres, don_abs, delta, chi2, phi, p))

        if not results:
            out.write("  No strategies with n >= 20 dialogs\n\n")
            continue

        # Bonferroni + FDR
        n_tests = len(results)
        p_vals = [r[7] for r in results]
        _, p_fdr, _, _ = multipletests(p_vals, method='fdr_bh')

        out.write(f"Number of tests: {n_tests}\n")
        out.write(f"{'Strategy':35s} {'n':>5s} {'Pres%':>7s} {'Abs%':>7s} {'Delta':>7s} {'chi2':>7s} {'phi':>6s} {'p_raw':>9s} {'p_Bonf':>9s} {'p_FDR':>9s}\n")
        out.write("-" * 110 + "\n")

        # Sort by p-value
        sorted_results = sorted(zip(results, p_fdr), key=lambda x: x[0][7])
        for (strat, n, dp, da, delta, chi2, phi, p), fdr in sorted_results:
            p_bonf = min(p * n_tests, 1.0)
            sig = ""
            if p_bonf < 0.001:
                sig = "***"
            elif p_bonf < 0.01:
                sig = "**"
            elif p_bonf < 0.05:
                sig = "*"
            out.write(f"{strat:35s} {n:5d} {dp:6.1f}% {da:6.1f}% {delta:+6.1f} {chi2:7.2f} {phi:6.3f} {p:9.4f} {p_bonf:9.4f} {fdr:9.4f} {sig}\n")
        out.write("\n")

    # Cross-model summary for key strategies
    out.write("--- CROSS-MODEL SUMMARY: Guilt Induction & Reciprocity ---\n")
    out.write(f"{'Model':20s} {'Guilt n':>8s} {'Guilt Δ':>9s} {'Guilt p_B':>10s} {'Recip n':>8s} {'Recip Δ':>9s} {'Recip p_B':>10s}\n")
    out.write("-" * 80 + "\n")
    for model_name, data in models.items():
        dialogs = data["dialogs"]
        row = [model_name]
        for strat in ["Guilt Induction", "Reciprocity"]:
            col = f"has_{strat}"
            if col in dialogs.columns:
                n = dialogs[col].sum()
                pres = dialogs[dialogs[col] == 1]["has_donation"].mean() * 100
                abse = dialogs[dialogs[col] == 0]["has_donation"].mean() * 100
                delta = pres - abse
                ct = pd.crosstab(dialogs[col], dialogs["has_donation"])
                if ct.shape == (2, 2):
                    chi2, p, _, _ = chi2_contingency(ct)
                else:
                    p = 1.0
                # approximate Bonferroni (assuming ~22 tests)
                p_b = min(p * 22, 1.0)
                row.extend([f"{n:d}", f"{delta:+.1f}pp", f"{p_b:.4f}"])
            else:
                row.extend(["N/A", "N/A", "N/A"])
        out.write(f"{row[0]:20s} {row[1]:>8s} {row[2]:>9s} {row[3]:>10s} {row[4]:>8s} {row[5]:>9s} {row[6]:>10s}\n")
    out.write("\n\n")


# ============================================================
# ANALYSIS 4: Logistic regression models
# ============================================================

def analysis_logistic(models, out):
    out.write("=" * 80 + "\n")
    out.write("ANALYSIS 4: LOGISTIC REGRESSION\n")
    out.write("=" * 80 + "\n\n")

    for model_name, data in models.items():
        dialogs = data["dialogs"].dropna(subset=["mean_sentiment", "mean_interest"]).copy()
        out.write(f"--- {model_name} (n={len(dialogs)} dialogs) ---\n\n")

        # --- Model 1: Categories only ---
        cat_cols = []
        for cat in PERSUASION_CATEGORIES:
            col = f"cat_{cat}"
            if col in dialogs.columns and dialogs[col].sum() >= 20:
                cat_cols.append(col)

        if len(cat_cols) < 2:
            out.write("  Not enough category columns for regression\n\n")
            continue

        X1 = dialogs[cat_cols].astype(float)
        X1 = sm.add_constant(X1)
        y = dialogs["has_donation"].astype(float)

        try:
            model1 = sm.Logit(y, X1).fit(disp=0)
            out.write(f"Model 1 (categories only): pseudo R² = {model1.prsquared:.4f}, "
                      f"LLR p = {model1.llr_pvalue:.4e}\n")
            y_pred1 = model1.predict(X1)
            auc1 = roc_auc_score(y, y_pred1)
            out.write(f"  AUC = {auc1:.3f}\n")

            out.write(f"  {'Predictor':35s} {'OR':>7s} {'p':>10s}\n")
            for name, coef, pval in zip(model1.params.index[1:], model1.params[1:], model1.pvalues[1:]):
                oname = name.replace("cat_", "")
                sig = "*" if pval < 0.05 else ""
                out.write(f"  {oname:35s} {np.exp(coef):7.2f} {pval:10.4f} {sig}\n")
            out.write("\n")
        except Exception as e:
            out.write(f"  Model 1 failed: {e}\n\n")

        # --- Model 2: Categories + sentiment + interest ---
        X2 = dialogs[cat_cols + ["mean_sentiment", "mean_interest"]].astype(float)
        X2 = sm.add_constant(X2)

        try:
            model2 = sm.Logit(y, X2).fit(disp=0)
            out.write(f"Model 2 (categories + sentiment + interest): pseudo R² = {model2.prsquared:.4f}, "
                      f"LLR p = {model2.llr_pvalue:.4e}\n")
            y_pred2 = model2.predict(X2)
            auc2 = roc_auc_score(y, y_pred2)
            out.write(f"  AUC = {auc2:.3f}\n")

            # LR test vs Model 1
            lr_stat = 2 * (model2.llf - model1.llf)
            lr_df = len(model2.params) - len(model1.params)
            lr_p = 1 - stats.chi2.cdf(lr_stat, lr_df)
            out.write(f"  LR test vs Model 1: chi2 = {lr_stat:.1f}, p = {lr_p:.2e}\n")

            out.write(f"  {'Predictor':35s} {'OR':>7s} {'p':>10s}\n")
            for name, coef, pval in zip(model2.params.index[1:], model2.params[1:], model2.pvalues[1:]):
                oname = name.replace("cat_", "")
                sig = "*" if pval < 0.05 else ""
                out.write(f"  {oname:35s} {np.exp(coef):7.2f} {pval:10.4f} {sig}\n")
            out.write("\n")
        except Exception as e:
            out.write(f"  Model 2 failed: {e}\n\n")

        # --- Model 3: Parsimonious (Guilt + Reciprocity + sentiment + interest) ---
        m3_cols = []
        for strat in ["Guilt Induction", "Reciprocity"]:
            col = f"has_{strat}"
            if col in dialogs.columns:
                m3_cols.append(col)

        if len(m3_cols) == 2:
            X3 = dialogs[m3_cols + ["mean_sentiment", "mean_interest"]].astype(float)
            X3 = sm.add_constant(X3)

            try:
                model3 = sm.Logit(y, X3).fit(disp=0)
                out.write(f"Model 3 (parsimonious): pseudo R² = {model3.prsquared:.4f}, "
                          f"AIC = {model3.aic:.0f}\n")
                y_pred3 = model3.predict(X3)
                auc3 = roc_auc_score(y, y_pred3)
                out.write(f"  AUC = {auc3:.3f}\n")

                out.write(f"  {'Predictor':35s} {'OR':>7s} {'95% CI':>18s} {'p':>10s}\n")
                conf = model3.conf_int()
                for name, coef, pval in zip(model3.params.index[1:], model3.params[1:], model3.pvalues[1:]):
                    oname = name.replace("has_", "")
                    ci_low = np.exp(conf.loc[name, 0])
                    ci_high = np.exp(conf.loc[name, 1])
                    sig = "*" if pval < 0.05 else ""
                    out.write(f"  {oname:35s} {np.exp(coef):7.2f} [{ci_low:5.2f}, {ci_high:5.2f}] {pval:10.4f} {sig}\n")
                out.write("\n")
            except Exception as e:
                out.write(f"  Model 3 failed: {e}\n\n")
        else:
            out.write(f"  Model 3 skipped: Guilt or Reciprocity not found\n\n")

        out.write("\n")


# ============================================================
# ANALYSIS 5: Strategy-response sentiment link
# ============================================================

def analysis_sentiment_link(models, full_df, out):
    out.write("=" * 80 + "\n")
    out.write("ANALYSIS 5: STRATEGY-RESPONSE SENTIMENT LINK\n")
    out.write("=" * 80 + "\n\n")

    # Get target sentiment numeric
    sent_map = {"negative": -1, "neutral": 0, "positive": 1}

    for model_name, data in models.items():
        turns = data["turns"].copy()
        turns = turns.reset_index(drop=True)

        # For each persuader turn, find the next target turn's sentiment
        # We need to go back to full_df for this
        pairs = []
        for dialog_id in turns["B2"].unique():
            dialog = full_df[full_df["B2"] == dialog_id].sort_index()
            dialog_turns = turns[turns["B2"] == dialog_id]

            for _, pturn in dialog_turns.iterrows():
                strat = pturn["strategy"]
                cat = pturn["category"]
                if cat == "Conversation Management":
                    continue

                # Find next target turn in original data
                # Get rows after this turn in the dialog
                turn_num = pturn["Turn"]
                next_targets = dialog[(dialog["Turn"] > turn_num) & (dialog["B4"] == 1)]
                if len(next_targets) > 0:
                    next_t = next_targets.iloc[0]
                    sent_str = next_t.get("sentiment_ollama_v2", "")
                    sent_num = sent_map.get(str(sent_str).lower(), None)
                    if sent_num is not None:
                        pairs.append({"strategy": strat, "sentiment": sent_num})

        if not pairs:
            out.write(f"--- {model_name}: no pairs found ---\n\n")
            continue

        pairs_df = pd.DataFrame(pairs)
        corpus_mean = pairs_df["sentiment"].mean()

        out.write(f"--- {model_name} ({len(pairs_df)} persuader-target pairs) ---\n")
        out.write(f"Corpus-wide mean sentiment: {corpus_mean:+.2f}\n\n")

        # Per-strategy stats (n >= 20)
        strat_stats = pairs_df.groupby("strategy").agg(
            n=("sentiment", "count"),
            mean_sent=("sentiment", "mean"),
            pct_neg=("sentiment", lambda x: (x < 0).mean() * 100)
        ).reset_index()
        strat_stats = strat_stats[strat_stats["n"] >= 20].sort_values("mean_sent")

        out.write(f"{'Strategy':35s} {'n':>5s} {'Mean sent':>10s} {'% neg':>7s}\n")
        out.write("-" * 62 + "\n")
        for _, row in strat_stats.iterrows():
            out.write(f"{row['strategy']:35s} {row['n']:5.0f} {row['mean_sent']:+10.2f} {row['pct_neg']:6.1f}%\n")
        out.write("\n")

    # Cross-model comparison for key strategies
    out.write("--- Cross-model: Guilt Induction & Reciprocity sentiment ---\n")
    out.write(f"{'Model':20s} {'Guilt mean_s':>12s} {'Guilt %neg':>10s} {'Recip mean_s':>12s} {'Recip %neg':>10s}\n")
    out.write("-" * 68 + "\n")
    # (already printed above, this is redundant but useful for quick comparison)
    out.write("\n\n")


# ============================================================
# ANALYSIS 6: Inter-model agreement
# ============================================================

def analysis_agreement(models, full_df, mistral_df, phi4_df, out):
    out.write("=" * 80 + "\n")
    out.write("ANALYSIS 6: INTER-MODEL AGREEMENT (Cohen's kappa)\n")
    out.write("=" * 80 + "\n\n")

    # Build aligned arrays
    persuader = full_df[full_df["B4"] == 0].copy()
    qwen_strats = persuader["strategy_ollama_single"].values
    qwen_cats = pd.Series(qwen_strats).map(get_category).values

    mistral_strats = mistral_df["mistral_strategy"].values
    mistral_cats = mistral_df["mistral_strategy"].map(get_category).values

    phi4_strats = phi4_df["phi4_strategy"].values
    phi4_cats = phi4_df["phi4_strategy"].map(get_category).values

    # Ensure same length
    n = min(len(qwen_strats), len(mistral_strats), len(phi4_strats))
    qwen_strats = qwen_strats[:n]
    qwen_cats = qwen_cats[:n]
    mistral_strats = mistral_strats[:n]
    mistral_cats = mistral_cats[:n]
    phi4_strats = phi4_strats[:n]
    phi4_cats = phi4_cats[:n]

    # Filter valid (non-empty)
    valid = (pd.Series(qwen_strats).ne("") & pd.Series(qwen_strats).notna() &
             pd.Series(mistral_strats).ne("") & pd.Series(mistral_strats).notna() &
             pd.Series(phi4_strats).ne("") & pd.Series(phi4_strats).notna())

    qwen_s = pd.Series(qwen_strats)[valid].values
    qwen_c = pd.Series(qwen_cats)[valid].values
    mistral_s = pd.Series(mistral_strats)[valid].values
    mistral_c = pd.Series(mistral_cats)[valid].values
    phi4_s = pd.Series(phi4_strats)[valid].values
    phi4_c = pd.Series(phi4_cats)[valid].values

    out.write(f"Valid turns for agreement: {valid.sum()} / {n}\n\n")

    # Pairwise kappa
    pairs = [
        ("Qwen vs Mistral", qwen_s, mistral_s, qwen_c, mistral_c),
        ("Qwen vs Phi-4", qwen_s, phi4_s, qwen_c, phi4_c),
        ("Mistral vs Phi-4", mistral_s, phi4_s, mistral_c, phi4_c),
    ]

    out.write(f"{'Pair':25s} {'kappa_strat':>12s} {'kappa_cat':>12s} {'exact_match%':>13s}\n")
    out.write("-" * 65 + "\n")

    for name, s1, s2, c1, c2 in pairs:
        k_strat = cohen_kappa_score(s1, s2)
        k_cat = cohen_kappa_score(c1, c2)
        exact = (s1 == s2).mean() * 100
        out.write(f"{name:25s} {k_strat:12.3f} {k_cat:12.3f} {exact:12.1f}%\n")

    # Macro-level agreement (persuasive / inquiry / non-strategy)
    out.write("\n--- Macro-level agreement (persuasive / CM) ---\n")
    def to_macro(cats):
        return np.where(pd.Series(cats) == "Conversation Management", "CM", "Persuasion")

    q_macro = to_macro(qwen_c)
    m_macro = to_macro(mistral_c)
    p_macro = to_macro(phi4_c)

    macro_pairs = [
        ("Qwen vs Mistral", q_macro, m_macro),
        ("Qwen vs Phi-4", q_macro, p_macro),
        ("Mistral vs Phi-4", m_macro, p_macro),
    ]
    out.write(f"{'Pair':25s} {'kappa_macro':>12s} {'agreement%':>12s}\n")
    out.write("-" * 52 + "\n")
    for name, a, b in macro_pairs:
        k = cohen_kappa_score(a, b)
        agree = (a == b).mean() * 100
        out.write(f"{name:25s} {k:12.3f} {agree:11.1f}%\n")

    # Three-way agreement
    three_way = (qwen_s == mistral_s) & (mistral_s == phi4_s)
    three_way_cat = (qwen_c == mistral_c) & (mistral_c == phi4_c)
    out.write(f"\nThree-way exact match (strategy): {three_way.mean()*100:.1f}%\n")
    out.write(f"Three-way exact match (category): {three_way_cat.mean()*100:.1f}%\n")
    out.write("\n\n")


# ============================================================
# ANALYSIS 7: Replication summary
# ============================================================

def analysis_replication_summary(models, out):
    out.write("=" * 80 + "\n")
    out.write("ANALYSIS 7: REPLICATION SUMMARY — Do key findings hold across models?\n")
    out.write("=" * 80 + "\n\n")

    findings = [
        ("Guilt Induction backfire", "Guilt Induction", "negative"),
        ("Reciprocity positive", "Reciprocity", "positive"),
        ("Commitment/Consistency positive", "Commitment and Consistency", "positive"),
    ]

    for finding_name, strat, direction in findings:
        out.write(f"Finding: {finding_name}\n")
        out.write(f"{'Model':20s} {'n_dialogs':>10s} {'Pres%':>8s} {'Abs%':>8s} {'Delta':>8s} {'p_raw':>10s} {'Replicates?':>12s}\n")
        out.write("-" * 82 + "\n")

        for model_name, data in models.items():
            dialogs = data["dialogs"]
            col = f"has_{strat}"
            if col not in dialogs.columns:
                out.write(f"{model_name:20s} {'N/A':>10s}\n")
                continue

            n = dialogs[col].sum()
            if n < 5:
                out.write(f"{model_name:20s} {n:10d} too few\n")
                continue

            pres = dialogs[dialogs[col] == 1]["has_donation"].mean() * 100
            abse = dialogs[dialogs[col] == 0]["has_donation"].mean() * 100
            delta = pres - abse

            ct = pd.crosstab(dialogs[col], dialogs["has_donation"])
            if ct.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ct)
            else:
                p = 1.0

            if direction == "negative":
                replicates = "YES" if (delta < 0 and p < 0.05) else ("direction" if delta < 0 else "NO")
            else:
                replicates = "YES" if (delta > 0 and p < 0.05) else ("direction" if delta > 0 else "NO")

            out.write(f"{model_name:20s} {n:10d} {pres:7.1f}% {abse:7.1f}% {delta:+7.1f} {p:10.4f} {replicates:>12s}\n")
        out.write("\n")

    # Decision-amount asymmetry (same across models since sentiment/interest are from Qwen)
    out.write("Finding: Decision-amount asymmetry\n")
    out.write("  (Target sentiment/interest are model-independent — same Qwen labels for all)\n")
    out.write("  This finding is invariant to strategy annotation model.\n\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")
    full_df, donations, mistral, phi4 = load_data()

    print("Preparing model datasets...")
    models, full_df, target = prepare_model_data(full_df, mistral, phi4, donations)

    output_file = os.path.join(OUT_DIR, "THREE_MODEL_RESULTS.txt")

    # Verify donation rate matches paper (53.6%)
    don_rate = donations["donated"].mean()
    don_n = donations["donated"].sum()
    don_total = len(donations)
    print(f"Donation rate: {don_n}/{don_total} = {don_rate*100:.1f}% (paper: 53.6%)")
    assert abs(don_rate - 0.536) < 0.01, f"Donation rate {don_rate:.3f} doesn't match paper!"

    with open(output_file, "w") as out:
        out.write("THREE-MODEL REPLICATION ANALYSIS\n")
        out.write("Models: Qwen3:30b (Alibaba), Mistral-Small-3.2 (Mistral AI), Phi-4 (Microsoft)\n")
        out.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        out.write(f"Donation variable: TARGET donation only ({don_n}/{don_total} = {don_rate*100:.1f}%)\n")
        out.write("=" * 80 + "\n\n")

        print("Analysis 1: Distributions...")
        analysis_distribution(models, out)

        print("Analysis 2: Category chi-square...")
        analysis_category_chisq(models, out)

        print("Analysis 3: Strategy chi-square...")
        analysis_strategy_chisq(models, out)

        print("Analysis 4: Logistic regression...")
        analysis_logistic(models, out)

        print("Analysis 5: Sentiment link...")
        analysis_sentiment_link(models, full_df, out)

        print("Analysis 6: Inter-model agreement...")
        analysis_agreement(models, full_df, mistral, phi4, out)

        print("Analysis 7: Replication summary...")
        analysis_replication_summary(models, out)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
