"""
Guilt Induction Intersection Analysis across three LLM annotators:
  Qwen (from full_dialog_with_all_analysis.csv)
  Mistral (from mistral_annotations.csv)
  Phi-4 (from phi4_annotations.csv)

For each model, identify dialogues containing at least one "Guilt Induction" turn.
Compute intersection sets and donation rates.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

# ── paths ──────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN_PATH    = os.path.join(_DIR, "full_dialog_with_all_analysis.csv")
MISTRAL_PATH = os.path.join(_DIR, "mistral_annotations.csv")
PHI4_PATH    = os.path.join(_DIR, "phi4_annotations.csv")
DONATION_PATH = os.path.join(_DIR, "sentiment_donation_by_dialog.csv")
OUT_PATH     = os.path.join(_DIR, "guilt_intersection_results.txt")

STRATEGY = "Guilt Induction"

# ── load data ──────────────────────────────────────────────────────────────────
qwen_df   = pd.read_csv(QWEN_PATH)
mistral_df = pd.read_csv(MISTRAL_PATH)
phi4_df    = pd.read_csv(PHI4_PATH)
donation_df = pd.read_csv(DONATION_PATH)

# ── identify Guilt-Induction dialogue sets ─────────────────────────────────────
# Qwen: column "strategy_ollama_single", persuader turns only (B4 even turn, role=0)
qwen_guilt = set(
    qwen_df.loc[
        qwen_df["strategy_ollama_single"].str.contains(STRATEGY, case=False, na=False),
        "B2"
    ].unique()
)

# Mistral: column "mistral_strategy"
mistral_guilt = set(
    mistral_df.loc[
        mistral_df["mistral_strategy"].str.contains(STRATEGY, case=False, na=False),
        "dialog_id"
    ].unique()
)

# Phi-4: column "phi4_strategy"
phi4_guilt = set(
    phi4_df.loc[
        phi4_df["phi4_strategy"].str.contains(STRATEGY, case=False, na=False),
        "dialog_id"
    ].unique()
)

print(f"Qwen  dialogues with Guilt Induction: {len(qwen_guilt)}")
print(f"Mistral dialogues with Guilt Induction: {len(mistral_guilt)}")
print(f"Phi-4 dialogues with Guilt Induction: {len(phi4_guilt)}")

# ── compute subsets ─────────────────────────────────────────────────────────────
intersection_all3 = qwen_guilt & mistral_guilt & phi4_guilt
qwen_only = qwen_guilt - mistral_guilt - phi4_guilt
mistral_only = mistral_guilt - qwen_guilt - phi4_guilt
phi4_only = phi4_guilt - qwen_guilt - mistral_guilt

any2 = set()
for d in qwen_guilt | mistral_guilt | phi4_guilt:
    count = sum([d in qwen_guilt, d in mistral_guilt, d in phi4_guilt])
    if count >= 2:
        any2.add(d)

any_guilt = qwen_guilt | mistral_guilt | phi4_guilt

# All dialogue IDs from the donation file
all_dialogs = set(donation_df["B2"].unique())
no_guilt = all_dialogs - any_guilt

# Also compute: exactly 2 agree (but not all 3)
exactly2 = any2 - intersection_all3

# Qwen & Mistral only (not Phi)
qwen_mistral = (qwen_guilt & mistral_guilt) - phi4_guilt
# Qwen & Phi only (not Mistral)
qwen_phi = (qwen_guilt & phi4_guilt) - mistral_guilt
# Mistral & Phi only (not Qwen)
mistral_phi = (mistral_guilt & phi4_guilt) - qwen_guilt

print(f"\nIntersection (all 3):  {len(intersection_all3)}")
print(f"At least 2 agree:     {len(any2)}")
print(f"Exactly 2 agree:      {len(exactly2)}")
print(f"Qwen only:            {len(qwen_only)}")
print(f"Mistral only:         {len(mistral_only)}")
print(f"Phi-4 only:           {len(phi4_only)}")
print(f"Any guilt (union):    {len(any_guilt)}")
print(f"No guilt at all:      {len(no_guilt)}")

# ── donation rates ──────────────────────────────────────────────────────────────
def donation_stats(dialog_set, label):
    """Compute donation rate for a set of dialogue IDs."""
    subset = donation_df[donation_df["B2"].isin(dialog_set)]
    n = len(subset)
    if n == 0:
        return {"label": label, "n": 0, "n_donated": 0, "donation_rate": np.nan}
    n_donated = subset["donated"].sum()
    rate = n_donated / n * 100
    return {"label": label, "n": int(n), "n_donated": int(n_donated), "donation_rate": rate}

results = []
results.append(donation_stats(intersection_all3, "All 3 agree (intersection)"))
results.append(donation_stats(any2, "At least 2 agree"))
results.append(donation_stats(exactly2, "Exactly 2 agree (not all 3)"))
results.append(donation_stats(qwen_only, "Qwen only"))
results.append(donation_stats(mistral_only, "Mistral only"))
results.append(donation_stats(phi4_only, "Phi-4 only"))
results.append(donation_stats(any_guilt, "Any guilt (union)"))
results.append(donation_stats(no_guilt, "No guilt in any model"))

# Pairwise overlaps
results.append(donation_stats(qwen_mistral, "Qwen & Mistral (not Phi-4)"))
results.append(donation_stats(qwen_phi, "Qwen & Phi-4 (not Mistral)"))
results.append(donation_stats(mistral_phi, "Mistral & Phi-4 (not Qwen)"))

print("\n" + "=" * 75)
print(f"{'Subset':<35} {'N':>5} {'Donated':>8} {'Rate%':>8}")
print("-" * 75)
for r in results:
    print(f"{r['label']:<35} {r['n']:>5} {r['n_donated']:>8} {r['donation_rate']:>7.1f}%")
print("=" * 75)

# ── statistical tests ──────────────────────────────────────────────────────────
def compare_groups(set_a, label_a, set_b, label_b):
    """Chi-square (or Fisher exact for small samples) comparing donation rates."""
    sub_a = donation_df[donation_df["B2"].isin(set_a)]
    sub_b = donation_df[donation_df["B2"].isin(set_b)]

    a_don = int(sub_a["donated"].sum())
    a_no  = len(sub_a) - a_don
    b_don = int(sub_b["donated"].sum())
    b_no  = len(sub_b) - b_don

    table = np.array([[a_don, a_no], [b_don, b_no]])

    # Fisher exact
    odds_ratio, p_fisher = fisher_exact(table)

    # Chi-square (with Yates correction)
    if table.min() >= 5:
        chi2, p_chi2, dof, expected = chi2_contingency(table, correction=True)
    else:
        chi2, p_chi2, dof, expected = chi2_contingency(table, correction=True)

    rate_a = a_don / (a_don + a_no) * 100 if (a_don + a_no) > 0 else 0
    rate_b = b_don / (b_don + b_no) * 100 if (b_don + b_no) > 0 else 0

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_a": a_don + a_no, "rate_a": rate_a,
        "n_b": b_don + b_no, "rate_b": rate_b,
        "chi2": chi2, "p_chi2": p_chi2,
        "odds_ratio": odds_ratio, "p_fisher": p_fisher,
        "table": table
    }

tests = []
tests.append(compare_groups(intersection_all3, "All-3-agree", no_guilt, "No guilt"))
tests.append(compare_groups(any2, "At-least-2", no_guilt, "No guilt"))
tests.append(compare_groups(any_guilt, "Any-guilt(union)", no_guilt, "No guilt"))
tests.append(compare_groups(qwen_only, "Qwen-only", no_guilt, "No guilt"))
tests.append(compare_groups(intersection_all3, "All-3-agree", qwen_only, "Qwen-only"))
tests.append(compare_groups(intersection_all3, "All-3-agree", any_guilt, "Any-guilt(union)"))

print("\n\n" + "=" * 100)
print("STATISTICAL TESTS (donation rate comparisons)")
print("=" * 100)
for t in tests:
    print(f"\n{t['comparison']}:")
    print(f"  Group A: n={t['n_a']}, donation rate={t['rate_a']:.1f}%")
    print(f"  Group B: n={t['n_b']}, donation rate={t['rate_b']:.1f}%")
    print(f"  Contingency table: {t['table'].tolist()}")
    print(f"  Chi-square = {t['chi2']:.4f}, p = {t['p_chi2']:.6f}")
    print(f"  Fisher exact: OR = {t['odds_ratio']:.4f}, p = {t['p_fisher']:.6f}")
    sig = "***" if t['p_fisher'] < 0.001 else "**" if t['p_fisher'] < 0.01 else "*" if t['p_fisher'] < 0.05 else "n.s."
    print(f"  Significance: {sig}")

# ── Jaccard similarities ───────────────────────────────────────────────────────
def jaccard(a, b):
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)

print("\n\n" + "=" * 75)
print("PAIRWISE JACCARD SIMILARITY (Guilt Induction dialogue sets)")
print("-" * 75)
print(f"  Qwen vs Mistral:  {jaccard(qwen_guilt, mistral_guilt):.4f}  (overlap: {len(qwen_guilt & mistral_guilt)})")
print(f"  Qwen vs Phi-4:    {jaccard(qwen_guilt, phi4_guilt):.4f}  (overlap: {len(qwen_guilt & phi4_guilt)})")
print(f"  Mistral vs Phi-4: {jaccard(mistral_guilt, phi4_guilt):.4f}  (overlap: {len(mistral_guilt & phi4_guilt)})")
print(f"  All-3 Jaccard:    {len(intersection_all3) / len(any_guilt):.4f}  (3-way)")
print("=" * 75)

# ── write results to file ──────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    f.write("GUILT INDUCTION INTERSECTION ANALYSIS\n")
    f.write("Three LLM annotators: Qwen, Mistral, Phi-4\n")
    f.write("=" * 75 + "\n\n")

    f.write("1. GUILT INDUCTION DIALOGUE COUNTS PER MODEL\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Qwen:    {len(qwen_guilt)} dialogues\n")
    f.write(f"  Mistral: {len(mistral_guilt)} dialogues\n")
    f.write(f"  Phi-4:   {len(phi4_guilt)} dialogues\n")
    f.write(f"  Total unique dialogues in donation data: {len(all_dialogs)}\n\n")

    f.write("2. SET INTERSECTIONS\n")
    f.write("-" * 50 + "\n")
    f.write(f"  All 3 agree (intersection):       {len(intersection_all3)}\n")
    f.write(f"  At least 2 agree:                  {len(any2)}\n")
    f.write(f"  Exactly 2 agree (not all 3):       {len(exactly2)}\n")
    f.write(f"  Qwen & Mistral (not Phi-4):        {len(qwen_mistral)}\n")
    f.write(f"  Qwen & Phi-4 (not Mistral):        {len(qwen_phi)}\n")
    f.write(f"  Mistral & Phi-4 (not Qwen):        {len(mistral_phi)}\n")
    f.write(f"  Qwen only:                         {len(qwen_only)}\n")
    f.write(f"  Mistral only:                      {len(mistral_only)}\n")
    f.write(f"  Phi-4 only:                        {len(phi4_only)}\n")
    f.write(f"  Any guilt (union):                 {len(any_guilt)}\n")
    f.write(f"  No guilt in any model:             {len(no_guilt)}\n\n")

    f.write("3. DONATION RATES BY SUBSET\n")
    f.write("-" * 75 + "\n")
    f.write(f"  {'Subset':<35} {'N':>5} {'Donated':>8} {'Rate%':>8}\n")
    f.write("  " + "-" * 60 + "\n")
    for r in results:
        f.write(f"  {r['label']:<35} {r['n']:>5} {r['n_donated']:>8} {r['donation_rate']:>7.1f}%\n")
    f.write("\n")

    f.write("4. STATISTICAL TESTS\n")
    f.write("=" * 100 + "\n")
    for t in tests:
        sig = "***" if t['p_fisher'] < 0.001 else "**" if t['p_fisher'] < 0.01 else "*" if t['p_fisher'] < 0.05 else "n.s."
        f.write(f"\n  {t['comparison']}:\n")
        f.write(f"    Group A: n={t['n_a']}, donation rate={t['rate_a']:.1f}%\n")
        f.write(f"    Group B: n={t['n_b']}, donation rate={t['rate_b']:.1f}%\n")
        f.write(f"    Contingency table: {t['table'].tolist()}\n")
        f.write(f"    Chi-square = {t['chi2']:.4f}, p = {t['p_chi2']:.6f}\n")
        f.write(f"    Fisher exact: OR = {t['odds_ratio']:.4f}, p = {t['p_fisher']:.6f}\n")
        f.write(f"    Significance: {sig}\n")

    f.write("\n\n5. PAIRWISE JACCARD SIMILARITY\n")
    f.write("-" * 75 + "\n")
    f.write(f"  Qwen vs Mistral:  {jaccard(qwen_guilt, mistral_guilt):.4f}  (overlap: {len(qwen_guilt & mistral_guilt)})\n")
    f.write(f"  Qwen vs Phi-4:    {jaccard(qwen_guilt, phi4_guilt):.4f}  (overlap: {len(qwen_guilt & phi4_guilt)})\n")
    f.write(f"  Mistral vs Phi-4: {jaccard(mistral_guilt, phi4_guilt):.4f}  (overlap: {len(mistral_guilt & phi4_guilt)})\n")
    f.write(f"  All-3 Jaccard:    {len(intersection_all3) / len(any_guilt):.4f}  (3-way: intersection/union)\n")

print(f"\nResults saved to {OUT_PATH}")
