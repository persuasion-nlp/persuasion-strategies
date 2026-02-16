#!/usr/bin/env python3
"""
Generate publication-quality figures for the LREC 2026 paper.
Outputs: fig_strategy_distribution.pdf, fig_guilt_sentiment.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Global style settings - sans-serif (Helvetica), matching LREC template
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'pdf.fonttype': 42,       # TrueType fonts in PDF (editable text)
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

import os
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================================================================
# FIGURE 1 - Strategy distribution (horizontal bar chart)
# ===================================================================

categories = [
    'Norms / Morality / Values',
    'Rational / Impact Appeal',
    'Framing & Presentation',
    'Authority / Expertise',
    'Emotional Influence',
    'Call to Action',
    'Commitment / Consistency',
    'Social Influence',
    'Exchange / Incentives',
    'Urgency / Scarcity',
    'Threat / Pressure',
]
counts = [1331, 973, 771, 697, 637, 646, 389, 247, 222, 19, 7]
total = 10600  # all annotated persuader turns (100% coverage)

# Sort ascending so highest count is at top of the chart
order = np.argsort(counts)
categories_sorted = [categories[i] for i in order]
counts_sorted = [counts[i] for i in order]

fig1, ax1 = plt.subplots(figsize=(3.5, 3.3))

bar_color = '#4682B4'  # steel blue
bars = ax1.barh(range(len(categories_sorted)), counts_sorted,
                color=bar_color, height=0.65, edgecolor='none')

ax1.set_yticks(range(len(categories_sorted)))
ax1.set_yticklabels(categories_sorted)
ax1.set_xlabel('Number of annotations')
ax1.set_xlim(0, max(counts_sorted) * 1.32)

# Count labels at end of each bar (with percentages)
for i, (bar, v) in enumerate(zip(bars, counts_sorted)):
    pct = v / total * 100
    ax1.text(v + 18, bar.get_y() + bar.get_height() / 2,
             f'{v} ({pct:.1f}%)', va='center', ha='left', fontsize=6.5)

ax1.tick_params(axis='y', length=0)  # no tick marks on y-axis
ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))

fig1.tight_layout(pad=0.3)
fig1.savefig(f'{OUT_DIR}/fig_strategy_distribution.pdf',
             bbox_inches='tight', pad_inches=0.02)
print('Saved fig_strategy_distribution.pdf')

# ===================================================================
# FIGURE 2 - Guilt induction & target sentiment (two-panel)
# ===================================================================

fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 2.5))

# --- Panel (a): Guilt induction effect on donation ---
guilt_labels = ['With Guilt\nInduction', 'Without Guilt\nInduction']
donated = [32.7, 56.0]
not_donated = [67.3, 44.0]

x_a = np.arange(len(guilt_labels))
w = 0.35

color_donated = '#2ca02c'      # green-ish for donated
color_not_donated = '#d62728'  # red-ish for not donated

bars_don = ax_a.bar(x_a - w/2, donated, w, label='Donated',
                    color=color_donated, edgecolor='none')
bars_not = ax_a.bar(x_a + w/2, not_donated, w, label='Not donated',
                    color=color_not_donated, edgecolor='none')

ax_a.set_ylabel('Percentage of conversations (%)')
ax_a.set_xticks(x_a)
ax_a.set_xticklabels(guilt_labels)
ax_a.set_ylim(0, 88)
ax_a.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax_a.legend(loc='upper right', frameon=False)
ax_a.set_title('(a) Guilt induction and donation rate', fontsize=8,
               fontweight='bold', pad=8)

# Value labels on bars
for bar in bars_don:
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
              f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=6.5)
for bar in bars_not:
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
              f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=6.5)

# Significance bracket
y_sig = 80
ax_a.plot([0, 0, 1, 1], [y_sig - 2, y_sig, y_sig, y_sig - 2],
          lw=0.8, color='black')
ax_a.text(0.5, y_sig + 0.5, '***p < 0.001', ha='center', va='bottom',
          fontsize=6.5)

# Sample size annotation
ax_a.text(0, -0.18, 'n = 104', ha='center', va='top', fontsize=6.5,
          transform=ax_a.get_xaxis_transform(), color='#555555')
ax_a.text(1, -0.18, 'n = 913', ha='center', va='top', fontsize=6.5,
          transform=ax_a.get_xaxis_transform(), color='#555555')

# --- Panel (b): Target sentiment and donation ---
sent_labels = ['Negative', 'Neutral', 'Positive']
donation_rates = [13.0, 47.3, 64.5]

# Sequential blue gradient (light -> dark)
sent_colors = ['#a6c8e0', '#4a90c4', '#1a5276']

x_b = np.arange(len(sent_labels))
bars_sent = ax_b.bar(x_b, donation_rates, 0.5, color=sent_colors,
                     edgecolor='none')

ax_b.set_ylabel('Donation rate (%)')
ax_b.set_xticks(x_b)
ax_b.set_xticklabels(sent_labels)
ax_b.set_ylim(0, 82)
ax_b.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax_b.set_title('(b) Target sentiment and donation', fontsize=8,
               fontweight='bold', pad=8)

# Value labels on bars
for bar, val in zip(bars_sent, donation_rates):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
              f'{val:.1f}%', ha='center', va='bottom', fontsize=6.5)

# Sample size annotations (consistent with panel a)
sample_sizes = [23, 577, 417]
for xi, n in zip(x_b, sample_sizes):
    ax_b.text(xi, -0.18, f'n = {n}', ha='center', va='top', fontsize=6.5,
              transform=ax_b.get_xaxis_transform(), color='#555555')

# Significance bracket spanning all three bars
y_sig2 = 74
ax_b.plot([0, 0, 2, 2], [y_sig2 - 2, y_sig2, y_sig2, y_sig2 - 2],
          lw=0.8, color='black')
ax_b.text(1, y_sig2 + 0.5, r'$\chi^2$(2), p < 0.001', ha='center', va='bottom',
          fontsize=6.5)

fig2.tight_layout(w_pad=2.5)
fig2.savefig(f'{OUT_DIR}/fig_guilt_sentiment.pdf',
             bbox_inches='tight', pad_inches=0.02)
print('Saved fig_guilt_sentiment.pdf')

plt.close('all')
print('Done - both figures generated.')
