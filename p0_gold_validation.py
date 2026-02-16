"""
P0: Gold Standard Validation
Compare student's LLM annotations (Qwen3:30b) against Wang et al. (2019)
gold annotations on the 300-dialogue PersuasionForGood AnnSet.

Granularity issue: Gold data is sentence-level, student data is turn-level.
Approach: aggregate gold to turn level, then map taxonomies and compute metrics.
"""

import pandas as pd
import numpy as np
from collections import Counter

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

gold = pd.read_excel('300_dialog.xlsx')  # from PersuasionForGood AnnSet
student = pd.read_csv('full_dialog_with_all_analysis.csv')

print(f"Gold: {gold.shape[0]} rows, {gold['B2'].nunique()} dialogs")
print(f"Student: {student.shape[0]} rows, {student['B2'].nunique()} dialogs")

# ============================================================
# 2. DEFINE MAPPING: Student 42 strategies -> Wang et al. labels
# ============================================================

# Wang et al. persuader labels (er_label_1):
# Persuasive Appeals:
#   logical-appeal, emotion-appeal, credibility-appeal,
#   foot-in-the-door, self-modeling, personal-story, donation-information
# Persuasive Inquiries:
#   source-related-inquiry, task-related-inquiry, personal-related-inquiry
# Non-strategy:
#   greeting, proposition-of-donation, acknowledgement, thank,
#   praise-user, ask-donation-amount, off-task, closing,
#   positive-to-inquiry, neutral-to-inquiry, negative-to-inquiry,
#   confirm-donation, ask-donate-more, comment-partner,
#   ask-not-donate-reason, you-are-welcome, other

STUDENT_TO_WANG = {
    # --- Persuasive Appeals ---

    # logical-appeal: use of reasoning, evidence, statistics
    "Rational Appeal": "logical-appeal",
    "Logical Appeal": "logical-appeal",

    # emotion-appeal: eliciting empathy, compassion, guilt, fear
    "Emotional Appeal": "emotion-appeal",
    "Empathy Appeal": "emotion-appeal",
    "Sympathy Appeal": "emotion-appeal",
    "Fear Appeal": "emotion-appeal",
    "Emotional Manipulation": "emotion-appeal",
    "Guilt Induction": "emotion-appeal",

    # credibility-appeal: credentials, organizational impact, transparency
    "Credibility Appeal": "credibility-appeal",
    "Authority": "credibility-appeal",
    "Expertise": "credibility-appeal",

    # foot-in-the-door: start small then escalate
    "Foot-in-the-door": "foot-in-the-door",
    "Door-in-the-face": "foot-in-the-door",  # related sequential request technique

    # self-modeling: persuader indicates own donation intention
    "Activation of Personal Commitment": "self-modeling",
    "Self-feeling Appeal": "self-modeling",

    # personal-story: personal narrative about donation experience
    "Storytelling": "personal-story",

    # donation-information: specific info about task, procedure, impact
    "Call to Action": "donation-information",
    "Framing": "donation-information",
    "Loss Aversion Appeal": "donation-information",

    # --- Persuasive Inquiries ---

    # source-related-inquiry: asks if persuadee knows the organization
    "Charity Awareness Probe": "source-related-inquiry",

    # task-related-inquiry: asks about persuadee's opinions
    "Qualification / Segmentation": "task-related-inquiry",
    "Donation Baseline / Habit Probe": "task-related-inquiry",
    "Permission / Time Check": "task-related-inquiry",

    # personal-related-inquiry: asks about personal experiences
    # (no direct equivalent in student's taxonomy)

    # --- Non-strategy labels ---
    "Greeting / Rapport": "greeting",
    "Acknowledgement": "acknowledgement",
    "Conversation Closing": "closing",
    "Non-persuasive Other": "other",
    "Logistics / Coordination": "proposition-of-donation",
    "Liking": "praise-user",

    # --- Strategies that map to OTHER (no clear Wang equivalent) ---
    "Moral Appeal": "emotion-appeal",       # moral duty -> emotional lever
    "Appeal to Values": "emotion-appeal",    # value-based -> emotional lever
    "Activation of Impersonal Commitment": "logical-appeal",  # normative argument
    "Commitment and Consistency": "logical-appeal",  # consistency principle
    "Social Proof": "credibility-appeal",    # others do it -> credibility
    "Unity": "credibility-appeal",           # group identity -> social credibility
    "Social Positioning": "credibility-appeal",
    "Urgency": "donation-information",       # urgency about the task
    "Scarcity": "donation-information",

    # --- Rare/aggressive strategies -> other ---
    "Rewarding Activity": "other",
    "Pre-giving": "other",
    "Reciprocity": "other",
    "Debt": "other",
    "Threat": "other",
    "Aversive Stimulation": "other",
    "Punishing Activity": "other",
    "Overloading": "other",
    "Confusion Induction": "other",
    "Bait-and-switch": "other",
    "Pretexting": "other",
    "Promise": "other",
    "Scarcity Manipulation": "other",
}

# Reverse mapping for reporting
WANG_STRATEGY_GROUPS = {
    "logical-appeal": ["Rational Appeal", "Logical Appeal", "Activation of Impersonal Commitment", "Commitment and Consistency"],
    "emotion-appeal": ["Emotional Appeal", "Empathy Appeal", "Sympathy Appeal", "Fear Appeal",
                        "Emotional Manipulation", "Guilt Induction", "Moral Appeal", "Appeal to Values"],
    "credibility-appeal": ["Credibility Appeal", "Authority", "Expertise", "Social Proof", "Unity", "Social Positioning"],
    "foot-in-the-door": ["Foot-in-the-door", "Door-in-the-face"],
    "self-modeling": ["Activation of Personal Commitment", "Self-feeling Appeal"],
    "personal-story": ["Storytelling"],
    "donation-information": ["Call to Action", "Framing", "Loss Aversion Appeal", "Urgency", "Scarcity"],
    "source-related-inquiry": ["Charity Awareness Probe"],
    "task-related-inquiry": ["Qualification / Segmentation", "Donation Baseline / Habit Probe", "Permission / Time Check"],
    "greeting": ["Greeting / Rapport"],
    "acknowledgement": ["Acknowledgement"],
    "closing": ["Conversation Closing"],
    "other": ["Non-persuasive Other", "Rewarding Activity", "Pre-giving", "Reciprocity", "Debt",
              "Threat", "Aversive Stimulation", "Punishing Activity", "Overloading",
              "Confusion Induction", "Bait-and-switch", "Pretexting", "Promise", "Scarcity Manipulation"],
    "proposition-of-donation": ["Logistics / Coordination"],
    "praise-user": ["Liking"],
}

# ============================================================
# 3. AGGREGATE GOLD DATA TO TURN LEVEL
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATING GOLD DATA TO TURN LEVEL")
print("=" * 70)

# Only persuader messages (B4=0) with labels
gold_er = gold[(gold['B4'] == 0) & (gold['er_label_1'].notna())].copy()
print(f"Gold persuader sentences with labels: {len(gold_er)}")

# Aggregate: for each (B2, Turn), collect all er_label_1 values
gold_turns = gold_er.groupby(['B2', 'Turn']).agg(
    labels=('er_label_1', list),
    texts=('Unit', list),
    primary_label=('er_label_1', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
    n_sentences=('er_label_1', 'count')
).reset_index()

print(f"Gold turns (aggregated): {len(gold_turns)}")
print(f"Turns with 1 sentence: {(gold_turns['n_sentences'] == 1).sum()}")
print(f"Turns with 2+ sentences: {(gold_turns['n_sentences'] > 1).sum()}")
print(f"Max sentences per turn: {gold_turns['n_sentences'].max()}")

# For multi-sentence turns, pick the most "informative" label
# Priority: persuasive appeals > inquiries > non-strategy
LABEL_PRIORITY = {
    'credibility-appeal': 10, 'logical-appeal': 10, 'emotion-appeal': 10,
    'donation-information': 10, 'personal-story': 10, 'self-modeling': 10,
    'foot-in-the-door': 10,
    'source-related-inquiry': 8, 'task-related-inquiry': 8, 'personal-related-inquiry': 8,
    'proposition-of-donation': 6, 'ask-donation-amount': 6, 'ask-donate-more': 6,
    'confirm-donation': 6,
    'praise-user': 4, 'comment-partner': 4,
    'positive-to-inquiry': 3, 'neutral-to-inquiry': 3, 'negative-to-inquiry': 3,
    'acknowledgement': 2, 'thank': 2, 'you-are-welcome': 2,
    'greeting': 1, 'closing': 1, 'off-task': 0, 'other': 0,
    'ask-not-donate-reason': 5,
}

def pick_best_label(labels):
    """Pick the most informative label from a list (for multi-sentence turns)."""
    if len(labels) == 1:
        return labels[0]
    scored = [(l, LABEL_PRIORITY.get(l, 0)) for l in labels]
    scored.sort(key=lambda x: -x[1])
    return scored[0][0]

gold_turns['best_label'] = gold_turns['labels'].apply(pick_best_label)

print("\nGold turn-level label distribution:")
print(gold_turns['best_label'].value_counts().to_string())

# ============================================================
# 4. MATCH WITH STUDENT DATA
# ============================================================

print("\n" + "=" * 70)
print("MATCHING STUDENT DATA WITH GOLD")
print("=" * 70)

# Student: only persuader messages (B4=0) with strategy
student_er = student[(student['B4'] == 0) & (student['strategy_ollama_single'].notna())].copy()
student_er = student_er[student_er['strategy_ollama_single'] != ''].copy()
print(f"Student persuader messages with strategies: {len(student_er)}")

# Filter to 300 gold dialogs only
gold_dialog_ids = set(gold['B2'].unique())
student_300 = student_er[student_er['B2'].isin(gold_dialog_ids)].copy()
print(f"Student messages in gold 300 dialogs: {len(student_300)}")

# Map student strategies to Wang labels
student_300['wang_label'] = student_300['strategy_ollama_single'].map(STUDENT_TO_WANG)
unmapped = student_300[student_300['wang_label'].isna()]['strategy_ollama_single'].unique()
if len(unmapped) > 0:
    print(f"\nWARNING: Unmapped student strategies: {unmapped}")
    student_300.loc[student_300['wang_label'].isna(), 'wang_label'] = 'other'

print(f"\nStudent mapped label distribution:")
print(student_300['wang_label'].value_counts().to_string())

# Merge on (B2, Turn)
merged = pd.merge(
    gold_turns[['B2', 'Turn', 'best_label', 'labels', 'n_sentences']],
    student_300[['B2', 'Turn', 'strategy_ollama_single', 'wang_label']],
    on=['B2', 'Turn'],
    how='inner'
)

print(f"\nMatched turns: {len(merged)}")
print(f"Unmatched gold turns: {len(gold_turns) - len(merged)}")
print(f"Unmatched student turns: {len(student_300) - len(merged)}")

# ============================================================
# 5. COMPUTE METRICS
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION METRICS")
print("=" * 70)

from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    accuracy_score, f1_score, precision_score, recall_score
)

gold_labels = merged['best_label'].values
pred_labels = merged['wang_label'].values

# --- 5a: ALL LABELS (including non-strategy) ---
print("\n--- ALL LABELS (including non-strategy) ---")
print(f"Total matched turns: {len(merged)}")
print(f"Accuracy: {accuracy_score(gold_labels, pred_labels):.3f}")
print(f"Cohen's kappa: {cohen_kappa_score(gold_labels, pred_labels):.3f}")
print(f"Macro F1: {f1_score(gold_labels, pred_labels, average='macro', zero_division=0):.3f}")
print(f"Weighted F1: {f1_score(gold_labels, pred_labels, average='weighted', zero_division=0):.3f}")

print("\nClassification Report (all labels):")
print(classification_report(gold_labels, pred_labels, zero_division=0))

# --- 5b: ONLY 10 PERSUASION STRATEGIES ---
print("\n--- 10 PERSUASION STRATEGIES ONLY ---")

strategy_labels = [
    'logical-appeal', 'emotion-appeal', 'credibility-appeal',
    'foot-in-the-door', 'self-modeling', 'personal-story',
    'donation-information',
    'source-related-inquiry', 'task-related-inquiry', 'personal-related-inquiry'
]

# Filter to rows where GOLD label is one of the 10 strategies
mask_gold_strategy = merged['best_label'].isin(strategy_labels)
# Also filter to rows where PREDICTED label is one of the 10 strategies
mask_both_strategy = mask_gold_strategy & merged['wang_label'].isin(strategy_labels)

merged_strat = merged[mask_gold_strategy].copy()
print(f"Turns with gold strategy label: {len(merged_strat)}")

gold_strat = merged_strat['best_label'].values
pred_strat = merged_strat['wang_label'].values

print(f"Accuracy: {accuracy_score(gold_strat, pred_strat):.3f}")
print(f"Cohen's kappa: {cohen_kappa_score(gold_strat, pred_strat):.3f}")
print(f"Macro F1: {f1_score(gold_strat, pred_strat, average='macro', zero_division=0):.3f}")
print(f"Weighted F1: {f1_score(gold_strat, pred_strat, average='weighted', zero_division=0):.3f}")

print("\nClassification Report (10 strategies):")
print(classification_report(gold_strat, pred_strat, labels=strategy_labels, zero_division=0))

# --- 5c: RELAXED MATCHING (match against ANY gold sentence label, not just best) ---
print("\n--- RELAXED MATCHING (pred matches ANY gold sentence label in turn) ---")

def relaxed_match(row):
    """Check if predicted label matches ANY of the gold labels for this turn."""
    return row['wang_label'] in row['labels']

merged['relaxed_match'] = merged.apply(relaxed_match, axis=1)
relaxed_acc = merged['relaxed_match'].mean()
print(f"Relaxed accuracy (all labels): {relaxed_acc:.3f}")

merged_strat['relaxed_match'] = merged_strat.apply(relaxed_match, axis=1)
relaxed_acc_strat = merged_strat['relaxed_match'].mean()
print(f"Relaxed accuracy (10 strategies only): {relaxed_acc_strat:.3f}")

# --- 5d: COARSE-GRAINED (3 macro-categories) ---
print("\n--- COARSE-GRAINED: 3 MACRO-CATEGORIES ---")

COARSE_MAP = {
    'logical-appeal': 'appeal', 'emotion-appeal': 'appeal',
    'credibility-appeal': 'appeal', 'foot-in-the-door': 'appeal',
    'self-modeling': 'appeal', 'personal-story': 'appeal',
    'donation-information': 'appeal',
    'source-related-inquiry': 'inquiry', 'task-related-inquiry': 'inquiry',
    'personal-related-inquiry': 'inquiry',
    'greeting': 'non-strategy', 'acknowledgement': 'non-strategy',
    'closing': 'non-strategy', 'other': 'non-strategy',
    'thank': 'non-strategy', 'off-task': 'non-strategy',
    'proposition-of-donation': 'non-strategy', 'praise-user': 'non-strategy',
    'positive-to-inquiry': 'non-strategy', 'neutral-to-inquiry': 'non-strategy',
    'negative-to-inquiry': 'non-strategy', 'confirm-donation': 'non-strategy',
    'ask-donation-amount': 'non-strategy', 'ask-donate-more': 'non-strategy',
    'comment-partner': 'non-strategy', 'you-are-welcome': 'non-strategy',
    'ask-not-donate-reason': 'non-strategy',
}

merged['gold_coarse'] = merged['best_label'].map(COARSE_MAP)
merged['pred_coarse'] = merged['wang_label'].map(COARSE_MAP)

# Handle any unmapped
merged['gold_coarse'] = merged['gold_coarse'].fillna('non-strategy')
merged['pred_coarse'] = merged['pred_coarse'].fillna('non-strategy')

print(f"Coarse accuracy: {accuracy_score(merged['gold_coarse'], merged['pred_coarse']):.3f}")
print(f"Coarse Cohen's kappa: {cohen_kappa_score(merged['gold_coarse'], merged['pred_coarse']):.3f}")
print(f"Coarse Macro F1: {f1_score(merged['gold_coarse'], merged['pred_coarse'], average='macro'):.3f}")
print("\nCoarse Classification Report:")
print(classification_report(merged['gold_coarse'], merged['pred_coarse']))

# ============================================================
# 6. CONFUSION ANALYSIS: MOST COMMON ERRORS
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS: TOP-20 MOST COMMON CONFUSIONS")
print("=" * 70)

errors = merged[merged['best_label'] != merged['wang_label']].copy()
print(f"Total errors: {len(errors)} / {len(merged)} ({len(errors)/len(merged)*100:.1f}%)")

error_pairs = errors.groupby(['best_label', 'wang_label']).size().reset_index(name='count')
error_pairs = error_pairs.sort_values('count', ascending=False).head(20)

print(f"\n{'Gold Label':<30} {'Predicted Label':<30} {'Count':>6}")
print("-" * 70)
for _, row in error_pairs.iterrows():
    print(f"{row['best_label']:<30} {row['wang_label']:<30} {row['count']:>6}")

# ============================================================
# 7. PER-STRATEGY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PER-STRATEGY: WHAT STUDENT PREDICTS FOR EACH GOLD LABEL")
print("=" * 70)

for label in sorted(merged['best_label'].unique()):
    subset = merged[merged['best_label'] == label]
    pred_dist = subset['wang_label'].value_counts()
    correct = pred_dist.get(label, 0)
    total = len(subset)
    acc = correct / total if total > 0 else 0
    print(f"\nGold: {label} (n={total}, acc={acc:.1%})")
    for pred, cnt in pred_dist.head(5).items():
        marker = " <-- CORRECT" if pred == label else ""
        print(f"  -> {pred}: {cnt} ({cnt/total:.1%}){marker}")

# ============================================================
# 8. SUMMARY TABLE FOR PAPER
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY TABLE FOR PAPER")
print("=" * 70)

print(f"\n{'Evaluation':<45} {'Value':>10}")
print("-" * 57)
print(f"{'Matched turns (gold ∩ student)':<45} {len(merged):>10}")
print(f"{'--- All labels ---':<45} {'':>10}")
print(f"{'  Accuracy':<45} {accuracy_score(gold_labels, pred_labels):>10.3f}")
print(f"{'  Cohen kappa':<45} {cohen_kappa_score(gold_labels, pred_labels):>10.3f}")
print(f"{'  Macro F1':<45} {f1_score(gold_labels, pred_labels, average='macro', zero_division=0):>10.3f}")
print(f"{'  Weighted F1':<45} {f1_score(gold_labels, pred_labels, average='weighted', zero_division=0):>10.3f}")
print(f"{'  Relaxed accuracy (any sentence match)':<45} {relaxed_acc:>10.3f}")
print(f"{'--- 10 strategies only ---':<45} {'':>10}")
print(f"{'  Accuracy':<45} {accuracy_score(gold_strat, pred_strat):>10.3f}")
print(f"{'  Cohen kappa':<45} {cohen_kappa_score(gold_strat, pred_strat):>10.3f}")
print(f"{'  Macro F1':<45} {f1_score(gold_strat, pred_strat, average='macro', zero_division=0):>10.3f}")
print(f"{'  Weighted F1':<45} {f1_score(gold_strat, pred_strat, average='weighted', zero_division=0):>10.3f}")
print(f"{'  Relaxed accuracy':<45} {relaxed_acc_strat:>10.3f}")
print(f"{'--- Coarse (appeal/inquiry/non-strategy) ---':<45} {'':>10}")
print(f"{'  Accuracy':<45} {accuracy_score(merged['gold_coarse'], merged['pred_coarse']):>10.3f}")
print(f"{'  Cohen kappa':<45} {cohen_kappa_score(merged['gold_coarse'], merged['pred_coarse']):>10.3f}")
print(f"{'  Macro F1':<45} {f1_score(merged['gold_coarse'], merged['pred_coarse'], average='macro'):>10.3f}")

# Save merged data for further analysis
merged.to_csv('gold_validation_merged.csv', index=False)
print(f"\nMerged data saved to gold_validation_merged.csv")
