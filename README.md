# How Much Does Persuasion Strategy Matter? Evidence from Charitable Donation Dialogues

Code and data for the paper submitted to **SoCon-NLPSI Workshop @ LREC-COLING 2026**.

## Overview

This repository contains the annotated dataset and analysis scripts for a study of persuasion strategy effectiveness in the [PersuasionForGood](https://gitlab.com/ucdavisnlp/persuasionforgood) corpus (Wang et al., 2019). We annotate all 1,017 dialogues (20,932 turns) with a taxonomy of 41 persuasion strategies in 11 categories, using three open-source LLMs from different model families: **Qwen3:30b** (primary annotator), **Mistral-Small-3.2**, and **Phi-4**.

## Repository Structure

### Data

| File | Description |
|---|---|
| `full_dialog_with_all_analysis.csv` | Main annotated dataset: 20,932 turns (10,600 persuader + 10,332 target) across 1,017 dialogues. Includes Qwen3:30b strategy labels, target sentiment, and interest scores. |
| `mistral_annotations.csv` | Mistral-Small-3.2 strategy annotations for all 10,600 persuader turns. |
| `phi4_annotations.csv` | Phi-4 strategy annotations for all 10,600 persuader turns. |
| `sentiment_donation_by_dialog.csv` | Dialogue-level target sentiment aggregation with donation outcomes (target donations only). |
| `donation_dataset_stats.csv` | Dialogue-level donation statistics. |
| `gold_validation_merged.csv` | Cross-taxonomy validation: LLM annotations mapped to Wang et al. (2019) gold labels for the 300-dialogue AnnSet. |

### Annotation Scripts (with prompt templates)

| File | Description |
|---|---|
| `annotate_qwen.py` | Annotation script for Qwen3:30b via Ollama. Contains the full hierarchical prompt template (system message + user message with taxonomy definitions). |
| `annotate_mistral.py` | Annotation script for Mistral-Small-3.2. Same prompt template, different model. |
| `annotate_phi4.py` | Annotation script for Phi-4. Same prompt template, different model. |

### Analysis Scripts

| File | Description |
|---|---|
| `three_model_analysis.py` | Main analysis: replicates all paper results across all three annotators. Strategy/category distributions, chi-square tests with Bonferroni and FDR corrections, Mann-Whitney U tests, logistic regression (Models 1-3), strategy-response sentiment link, inter-model agreement. |
| `guilt_intersection_analysis.py` | Guilt Induction intersection analysis: set overlaps across three models, donation rates by subset, statistical tests. |
| `compute_supplementary_stats.py` | Supplementary statistics: sentiment-link pair breakdown, strategies per dialogue, category coverage. |
| `revised_statistical_analysis.py` | Single-model (Qwen) statistical analysis with full bivariate and multivariate tests. |
| `p0_gold_validation.py` | Cross-taxonomy validation against Wang et al. (2019) gold annotations. Requires `300_dialog.xlsx` from the [PersuasionForGood AnnSet](https://gitlab.com/ucdavisnlp/persuasionforgood). |
| `generate_figures.py` | Generates publication-quality PDF figures: strategy distribution and guilt/sentiment panels. |

## Requirements

```
pip install pandas numpy scipy statsmodels scikit-learn matplotlib
```

Python 3.8+.

## Reproducing Results

1. Clone this repository.
2. Run the three-model analysis (reproduces all main paper results):
   ```
   python three_model_analysis.py
   ```
3. Run the guilt intersection analysis:
   ```
   python guilt_intersection_analysis.py
   ```
4. Run the gold standard validation (requires `300_dialog.xlsx` from [PersuasionForGood](https://gitlab.com/ucdavisnlp/persuasionforgood)):
   ```
   python p0_gold_validation.py
   ```
5. Generate figures:
   ```
   python generate_figures.py
   ```

## Data Format

### Main dataset (`full_dialog_with_all_analysis.csv`)

| Column | Description |
|---|---|
| `Unit` | Turn text |
| `Turn` | Turn number within dialogue |
| `B4` | Speaker role: 0 = persuader, 1 = target |
| `B2` | Dialogue ID |
| `strategy_ollama_single` | Assigned persuasion strategy (41 labels, persuader turns only) |
| `sentiment_ollama_v2` | Target sentiment: negative / neutral / positive (target turns only) |
| `interest_ollama_v2` | Interest in donation score (target turns only) |
| `interest_label_ollama_v2` | Interest label: low / medium / high (target turns only) |

### Annotation CSVs (`mistral_annotations.csv`, `phi4_annotations.csv`)

| Column | Description |
|---|---|
| `dialog_id` | Dialogue ID (corresponds to `B2` in the main dataset) |
| `turn` | Turn number (corresponds to `Turn` in the main dataset) |
| `utterance` | Turn text (corresponds to `Unit` in the main dataset) |
| `mistral_strategy` / `phi4_strategy` | Assigned persuasion strategy (41 labels) |
| `mistral_category` / `phi4_category` | Assigned persuasion category (11 categories + Conversation Management) |
| `qwen_strategy` | Qwen3:30b strategy for cross-model comparison |
| `original_index` | Row index in the main dataset |

## Annotation Methodology

All three models use the same two-step hierarchical prompt:
1. The system message instructs the model to act as a hierarchical persuasion strategy classification system.
2. The user message presents (a) the persuader utterance to classify, (b) up to 5 previous dialogue turns as context, and (c) the full category-to-strategy hierarchy with definitions.
3. The model selects a parent category, then a specific strategy, returning structured JSON.
4. Temperature is set to 0.1 for near-deterministic output.

See `annotate_qwen.py` for the complete prompt template.

## Citation

```bibtex
@inproceedings{anonymous2026persuasion,
  title={How Much Does Persuasion Strategy Matter? Evidence from Charitable Donation Dialogues},
  author={Anonymous},
  booktitle={Proceedings of the SoCon-NLPSI Workshop at LREC-COLING 2026},
  year={2026}
}
```

## License

The original PersuasionForGood corpus is distributed under its [original license](https://gitlab.com/ucdavisnlp/persuasionforgood). Our annotations and analysis code are released for research purposes.
