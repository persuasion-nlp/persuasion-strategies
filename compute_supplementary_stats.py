"""
Compute supplementary statistics for the paper from PersuasionForGood annotation data.

Tasks:
A. Explain the 5,387 sentiment-link pairs
B. Unique workers (from dialog IDs)
C. Strategies per dialogue statistics
D. Percentage of dialogues containing each category
"""

import pandas as pd
import numpy as np
import os
import sys

# ============================================================
# PATHS
# ============================================================

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_DIR, "full_dialog_with_all_analysis.csv")
DONATION_PATH = os.path.join(_DIR, "sentiment_donation_by_dialog.csv")
OUT_PATH = os.path.join(_DIR, "supplementary_stats.txt")

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

CM_STRATEGIES = [k for k, v in STRATEGY_TO_CATEGORY.items() if v == "Conversation Management"]

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
donations = pd.read_csv(DONATION_PATH)

# Add category column
df["category"] = df["strategy_ollama_single"].map(STRATEGY_TO_CATEGORY)

# ============================================================
# OUTPUT
# ============================================================

lines = []


def out(text=""):
    lines.append(text)
    print(text)


out("=" * 80)
out("SUPPLEMENTARY STATISTICS FOR PAPER")
out("=" * 80)

# ============================================================
# A. EXPLAIN THE 5,387 SENTIMENT-LINK PAIRS
# ============================================================

out("\n" + "=" * 80)
out("A. EXPLAINING THE 5,387 SENTIMENT-LINK PAIRS")
out("=" * 80)

# A.1: Total persuader turns
persuader = df[df["B4"] == 0]
target = df[df["B4"] == 1]
n_persuader_total = len(persuader)
out(f"\nA.1. Total persuader turns (B4 == 0): {n_persuader_total:,}")

# A.2: Persuader turns that are Conversation Management
persuader_cm = persuader[persuader["category"] == "Conversation Management"]
n_cm = len(persuader_cm)
out(f"A.2. Persuader turns classified as Conversation Management: {n_cm:,}")

# How many have no strategy assigned (NaN)?
persuader_no_strategy = persuader[persuader["strategy_ollama_single"].isna()]
n_no_strategy = len(persuader_no_strategy)
out(f"     Persuader turns with no strategy assigned (NaN): {n_no_strategy:,}")

# A.3: Persuasion-strategy turns (non-CM persuader turns with a valid strategy)
persuader_with_strategy = persuader[persuader["strategy_ollama_single"].notna()]
persuader_persuasion = persuader_with_strategy[persuader_with_strategy["category"] != "Conversation Management"]
n_persuasion = len(persuader_persuasion)
out(f"A.3. Persuasion-strategy turns (non-CM, has strategy): {n_persuasion:,}")

# A.4: Of those, count how many have an immediately following target turn in the same dialogue
#
# IMPORTANT: The paper's code (three_model_analysis.py, line 556) uses:
#   next_targets = dialog[(dialog["Turn"] > turn_num) & (dialog["B4"] == 1)]
# This means the pairing is: for a persuader at Turn T, the NEXT target turn
# is at Turn > T (strictly greater), i.e., the target's response in the NEXT
# exchange, not the same exchange.
#
# The data structure is:
#   Turn 0, B4=0 -> persuader's 1st message
#   Turn 0, B4=1 -> target's 1st response
#   Turn 1, B4=0 -> persuader's 2nd message
#   Turn 1, B4=1 -> target's 2nd response
#   ...
# So the paper pairs persuader at Turn T with target at Turn T+1 (next exchange).

# Sort to ensure ordering
df_sorted = df.sort_values(["B2", "Unnamed: 0"]).reset_index(drop=True)

sent_map = {"negative": -1, "neutral": 0, "positive": 1}

# METHOD 1: Paper's method (Turn > turn_num) - should give 5,387
pair_count_paper = 0
for idx in persuader_persuasion.index:
    row = df.loc[idx]
    dialog_id = row["B2"]
    turn_num = row["Turn"]
    # Find next target turn with Turn > current turn_num (paper's method)
    next_targets = df_sorted[(df_sorted["B2"] == dialog_id) &
                              (df_sorted["Turn"] > turn_num) &
                              (df_sorted["B4"] == 1)]
    if len(next_targets) > 0:
        next_t = next_targets.iloc[0]
        sent_str = str(next_t.get("sentiment_ollama_v2", "")).lower()
        sent_num = sent_map.get(sent_str, None)
        if sent_num is not None:
            pair_count_paper += 1

out(f"A.4a. PAPER METHOD: Persuader at Turn T paired with next target at Turn > T")
out(f"      (matches three_model_analysis.py logic): {pair_count_paper:,}")

# METHOD 2: Same-turn method (Turn == turn_num) - the alternative interpretation
pair_count_same = 0
for idx in persuader_persuasion.index:
    row = df.loc[idx]
    dialog_id = row["B2"]
    turn_num = row["Turn"]
    # Find target turn at same Turn number
    same_targets = df_sorted[(df_sorted["B2"] == dialog_id) &
                              (df_sorted["Turn"] == turn_num) &
                              (df_sorted["B4"] == 1)]
    if len(same_targets) > 0:
        target_row = same_targets.iloc[0]
        sent_str = str(target_row.get("sentiment_ollama_v2", "")).lower()
        sent_num = sent_map.get(sent_str, None)
        if sent_num is not None:
            pair_count_same += 1

out(f"A.4b. SAME-TURN METHOD: Persuader at Turn T paired with target at Turn T")
out(f"      (same exchange): {pair_count_same:,}")

out(f"\nA.5. VERIFICATION: Does the paper method equal 5,387?")
out(f"     Paper method (Turn > T):  {pair_count_paper:,}  {'MATCHES' if pair_count_paper == 5387 else 'DOES NOT MATCH'}")
out(f"     Same-turn (Turn == T):    {pair_count_same:,}")

# Count the reasons for exclusion in the paper method
n_no_next_target = 0
n_no_sentiment = 0
for idx in persuader_persuasion.index:
    row = df.loc[idx]
    dialog_id = row["B2"]
    turn_num = row["Turn"]
    next_targets = df_sorted[(df_sorted["B2"] == dialog_id) &
                              (df_sorted["Turn"] > turn_num) &
                              (df_sorted["B4"] == 1)]
    if len(next_targets) == 0:
        n_no_next_target += 1
    else:
        next_t = next_targets.iloc[0]
        sent_str = str(next_t.get("sentiment_ollama_v2", "")).lower()
        sent_num = sent_map.get(sent_str, None)
        if sent_num is None:
            n_no_sentiment += 1

out(f"\nA.6. BREAKDOWN (paper method):")
out(f"     Total persuader turns:                       {n_persuader_total:>6,}")
out(f"     - No strategy assigned (NaN):                {n_no_strategy:>6,}")
out(f"     - Conversation Management:                   {n_cm:>6,}")
out(f"     = Persuasion-strategy turns:                 {n_persuasion:>6,}")
out(f"     - No next target turn (Turn > T):            {n_no_next_target:>6,}")
out(f"     - Next target exists but no sentiment:       {n_no_sentiment:>6,}")
out(f"     = Sentiment-link pairs (paper method):       {pair_count_paper:>6,}")

# ============================================================
# B. UNIQUE WORKERS
# ============================================================

out("\n" + "=" * 80)
out("B. UNIQUE WORKERS")
out("=" * 80)

# The B2 column contains dialog IDs like "20180904-045349_715_live"
# In the original PersuasionForGood dataset:
#   - B1 = persuader worker ID (AMT worker ID)
#   - B2 = dialog session ID
#   - B3 = target worker ID (AMT worker ID)
# Our dataset only has B2 (dialog ID). The number in the middle (e.g., 715)
# is NOT a unique worker ID - it's part of the dialog session identifier.

# Since we don't have B1/B3 columns, we cannot compute unique worker counts
# from this dataset alone. We note what the original PersuasionForGood paper reports.

n_dialogs = df["B2"].nunique()

out(f"\nNote: The released annotation data contains only dialog IDs (B2 column),")
out(f"not individual worker IDs (B1/B3). Unique worker counts cannot be computed")
out(f"from this dataset alone.")
out(f"")
out(f"The original PersuasionForGood paper (Wang et al., 2019) reports:")
out(f"  - 1,017 dialogues collected via Amazon Mechanical Turk")
out(f"  - Workers played either a persuader or target role")
out(f"  - The number in the dialog ID (e.g., '715' in '20180904-045349_715_live')")
out(f"    is a session/task identifier, not a unique worker ID.")
out(f"")
out(f"From our data:")
out(f"  N unique dialog IDs (B2): {n_dialogs:,}")

# Try to extract unique numeric IDs from B2 to give a rough count
# Format: YYYYMMDD-HHMMSS_NNN_live
numeric_ids = df["B2"].str.extract(r'_(\d+)_live')[0].dropna().unique()
out(f"  N unique numeric identifiers extracted from dialog IDs: {len(numeric_ids):,}")
out(f"  (These are session/task IDs, not necessarily unique worker IDs)")

# ============================================================
# C. STRATEGIES PER DIALOGUE
# ============================================================

out("\n" + "=" * 80)
out("C. DISTINCT PERSUASION STRATEGIES PER DIALOGUE")
out("=" * 80)

# For each dialogue, count the number of DISTINCT persuasion strategies used
# (excluding Conversation Management and NaN)

persuader_strats = df[(df["B4"] == 0) &
                      (df["strategy_ollama_single"].notna()) &
                      (df["category"] != "Conversation Management")]

strats_per_dialog = persuader_strats.groupby("B2")["strategy_ollama_single"].nunique()

# Some dialogues may have zero persuasion strategies (all CM)
# Include those as 0
all_dialogs = df["B2"].unique()
strats_per_dialog_full = strats_per_dialog.reindex(all_dialogs, fill_value=0)

mean_val = strats_per_dialog_full.mean()
median_val = strats_per_dialog_full.median()
std_val = strats_per_dialog_full.std()
min_val = strats_per_dialog_full.min()
max_val = strats_per_dialog_full.max()
q1 = strats_per_dialog_full.quantile(0.25)
q3 = strats_per_dialog_full.quantile(0.75)
iqr = q3 - q1

out(f"\nStatistics for distinct persuasion strategies per dialogue (N = {len(strats_per_dialog_full):,} dialogues):")
out(f"  Mean:   {mean_val:.2f}")
out(f"  Median: {median_val:.1f}")
out(f"  SD:     {std_val:.2f}")
out(f"  Min:    {min_val}")
out(f"  Max:    {max_val}")
out(f"  Q1:     {q1:.1f}")
out(f"  Q3:     {q3:.1f}")
out(f"  IQR:    {iqr:.1f}")

# Distribution
out(f"\nDistribution of distinct persuasion strategies per dialogue:")
value_counts = strats_per_dialog_full.value_counts().sort_index()
for n_strat, count in value_counts.items():
    pct = count / len(strats_per_dialog_full) * 100
    out(f"  {int(n_strat):>2} strategies: {count:>4} dialogues ({pct:5.1f}%)")

# ============================================================
# C2. DISTINCT PERSUASION CATEGORIES PER DIALOGUE
# ============================================================

out("\n" + "-" * 60)
out("C2. DISTINCT PERSUASION CATEGORIES PER DIALOGUE")
out("-" * 60)

persuader_cats = df[(df["B4"] == 0) &
                    (df["strategy_ollama_single"].notna()) &
                    (df["category"] != "Conversation Management")]

cats_per_dialog = persuader_cats.groupby("B2")["category"].nunique()
cats_per_dialog_full = cats_per_dialog.reindex(all_dialogs, fill_value=0)

mean_cat = cats_per_dialog_full.mean()
median_cat = cats_per_dialog_full.median()
std_cat = cats_per_dialog_full.std()
min_cat = cats_per_dialog_full.min()
max_cat = cats_per_dialog_full.max()
q1_cat = cats_per_dialog_full.quantile(0.25)
q3_cat = cats_per_dialog_full.quantile(0.75)
iqr_cat = q3_cat - q1_cat

out(f"\nStatistics for distinct persuasion categories per dialogue (N = {len(cats_per_dialog_full):,} dialogues):")
out(f"  Mean:   {mean_cat:.2f}")
out(f"  Median: {median_cat:.1f}")
out(f"  SD:     {std_cat:.2f}")
out(f"  Min:    {min_cat}")
out(f"  Max:    {max_cat}")
out(f"  Q1:     {q1_cat:.1f}")
out(f"  Q3:     {q3_cat:.1f}")
out(f"  IQR:    {iqr_cat:.1f}")

# ============================================================
# D. PERCENTAGE OF DIALOGUES CONTAINING EACH CATEGORY
# ============================================================

out("\n" + "=" * 80)
out("D. PERCENTAGE OF DIALOGUES CONTAINING EACH PERSUASION CATEGORY")
out("=" * 80)

out(f"\nTotal dialogues: {n_dialogs:,}")
out(f"\n{'Category':<30} {'N dialogs':>10} {'% of 1,017':>10}")
out("-" * 55)

category_pcts = {}
for cat in PERSUASION_CATEGORIES:
    # Dialogues that have at least one turn with this category
    dialogs_with_cat = persuader_strats[persuader_strats["category"] == cat]["B2"].nunique()
    pct = dialogs_with_cat / n_dialogs * 100
    category_pcts[cat] = (dialogs_with_cat, pct)
    out(f"{cat:<30} {dialogs_with_cat:>10,} {pct:>9.1f}%")

out("-" * 55)

# Verify paper claims
out(f"\nVerification against paper claims:")
norms_n, norms_pct = category_pcts.get("Norms / Morality / Values", (0, 0))
rational_n, rational_pct = category_pcts.get("Rational / Impact Appeal", (0, 0))
out(f"  Norms / Morality / Values: {norms_pct:.1f}% (paper claims ~71%)")
out(f"  Rational / Impact Appeal:  {rational_pct:.1f}% (paper claims ~54%)")

# Also show Conversation Management for completeness
cm_dialogs = persuader[persuader["category"] == "Conversation Management"]["B2"].nunique()
out(f"\n  Conversation Management present in: {cm_dialogs:,} / {n_dialogs:,} dialogues ({cm_dialogs/n_dialogs*100:.1f}%)")

# ============================================================
# ADDITIONAL: Strategy-level breakdown for reference
# ============================================================

out("\n" + "=" * 80)
out("ADDITIONAL: STRATEGY-LEVEL FREQUENCY (persuader turns only)")
out("=" * 80)

strat_counts = persuader["strategy_ollama_single"].value_counts()
out(f"\n{'Strategy':<40} {'Count':>6} {'% of persuader':>12}")
out("-" * 62)
for strat, count in strat_counts.items():
    cat = STRATEGY_TO_CATEGORY.get(strat, "???")
    pct = count / n_persuader_total * 100
    marker = " [CM]" if cat == "Conversation Management" else ""
    out(f"{strat:<40} {count:>6} {pct:>11.1f}%{marker}")
out(f"{'NaN (no strategy)':<40} {n_no_strategy:>6} {n_no_strategy/n_persuader_total*100:>11.1f}%")

# ============================================================
# SAVE
# ============================================================

with open(OUT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n\nResults saved to: {OUT_PATH}")
