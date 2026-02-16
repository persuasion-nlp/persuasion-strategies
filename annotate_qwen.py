"""
Re-annotate PersuasionForGood persuader turns using Qwen3:30b
via Ollama, replicating the original methodology
to achieve 100% coverage (original run had 97.7%).

Methodology (from paper):
- Two-step hierarchical prompt: category → strategy
- Up to 5 previous turns as context
- Temperature 0.1 for reproducibility
- Structured JSON output
- Only persuader turns (B4 == 0) are annotated
"""

import pandas as pd
import json
import requests
import time
import sys
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:30b"
TEMPERATURE = 0.1
CONTEXT_TURNS = 5

INPUT_CSV = os.path.join(os.path.dirname(__file__),
    "../../release/full_dialog_with_all_analysis.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),
    "qwen_annotations.csv")
CHECKPOINT_CSV = os.path.join(os.path.dirname(__file__),
    "qwen_checkpoint.csv")
LOG_FILE = os.path.join(os.path.dirname(__file__),
    "qwen_annotation.log")

# ============================================================
# TAXONOMY DEFINITION
# ============================================================

TAXONOMY = {
    "Norms / Morality / Values": {
        "description": "Appeals based on moral obligations, personal values, or ethical standards.",
        "strategies": {
            "Moral Appeal": "Invoking moral duty or ethical obligation to help others.",
            "Appeal to Values": "Connecting the request to the target's personal values or beliefs.",
            "Guilt Induction": "Making the target feel guilty for not helping or not caring enough.",
            "Self-feeling Appeal": "Appealing to how the target will feel about themselves (pride, self-worth) after donating.",
            "Activation of Impersonal Commitment": "Invoking abstract commitments to society or humanity rather than personal ones."
        }
    },
    "Rational / Impact Appeal": {
        "description": "Using logic, evidence, facts, or demonstrating concrete impact.",
        "strategies": {
            "Rational Appeal": "Presenting facts, statistics, or evidence about the cause or organization.",
            "Logical Appeal": "Using structured logical reasoning or cause-effect arguments."
        }
    },
    "Framing & Presentation": {
        "description": "How information is framed or presented to influence perception.",
        "strategies": {
            "Framing": "Presenting information in a particular frame to shape perception (e.g., emphasizing positive impact).",
            "Loss Aversion Appeal": "Emphasizing what will be lost if the target does not act.",
            "Bait-and-switch": "Starting with one topic then switching to the donation request.",
            "Pretexting": "Using a fabricated scenario to establish rapport before making the request."
        }
    },
    "Authority / Expertise": {
        "description": "Leveraging credibility, authority, or expert knowledge.",
        "strategies": {
            "Credibility Appeal": "Citing the credibility, reputation, or track record of the charity or cause.",
            "Authority": "Invoking authority figures or institutions to support the request.",
            "Expertise": "Demonstrating expert knowledge about the cause or domain."
        }
    },
    "Emotional Influence": {
        "description": "Directly engaging emotions to motivate action.",
        "strategies": {
            "Empathy Appeal": "Encouraging the target to empathize with those in need.",
            "Storytelling": "Using personal stories or narratives to create emotional engagement.",
            "Emotional Appeal": "General appeal to emotions (compassion, sadness, hope) without specific technique.",
            "Fear Appeal": "Highlighting frightening consequences of inaction.",
            "Sympathy Appeal": "Eliciting sympathy or pity for those in need.",
            "Emotional Manipulation": "Deliberate manipulation of emotions through exaggeration or distortion."
        }
    },
    "Call to Action": {
        "description": "Direct requests or prompts to take action.",
        "strategies": {
            "Call to Action": "Directly asking the target to donate or take a specific action.",
            "Liking": "Building rapport or expressing liking to make the target more receptive."
        }
    },
    "Commitment / Consistency": {
        "description": "Leveraging prior commitments or desire for consistency.",
        "strategies": {
            "Commitment and Consistency": "Referencing the target's prior statements or commitments to encourage follow-through.",
            "Activation of Personal Commitment": "Activating the target's personal sense of commitment to the cause.",
            "Foot-in-the-door": "Starting with a small request before making a larger one.",
            "Door-in-the-face": "Starting with a large request then retreating to a smaller one."
        }
    },
    "Social Influence": {
        "description": "Using social norms, group identity, or others' behavior.",
        "strategies": {
            "Social Proof": "Citing what others have done (e.g., 'many people donate', 'it's common').",
            "Unity": "Emphasizing shared group identity or togetherness ('we', 'us', 'together').",
            "Social Positioning": "Positioning donation as a way to gain social status or approval."
        }
    },
    "Exchange / Incentives": {
        "description": "Offering something in return or creating reciprocal obligation.",
        "strategies": {
            "Reciprocity": "Offering something or doing a favor first to create obligation to reciprocate.",
            "Rewarding Activity": "Highlighting rewards or positive outcomes of donating.",
            "Pre-giving": "Giving information, compliments, or assistance before making the request.",
            "Debt": "Suggesting the target owes something to the cause or persuader."
        }
    },
    "Urgency / Scarcity": {
        "description": "Creating time pressure or emphasizing limited opportunity.",
        "strategies": {
            "Urgency": "Emphasizing that action is needed immediately or soon.",
            "Scarcity": "Suggesting that the opportunity to help is limited or rare."
        }
    },
    "Threat / Pressure": {
        "description": "Using threats, pressure, or aggressive tactics.",
        "strategies": {
            "Threat": "Implying negative consequences for the target if they don't comply.",
            "Aversive Stimulation": "Persistent nagging or pressure until the target complies.",
            "Punishing Activity": "Suggesting punishment or negative treatment for non-compliance.",
            "Overloading": "Overwhelming the target with information or requests.",
            "Confusion Induction": "Deliberately confusing the target to gain compliance."
        }
    },
    "Conversation Management": {
        "description": "Non-persuasive dialogue management turns.",
        "strategies": {
            "Greeting / Rapport": "Greetings, small talk, rapport-building without persuasive intent.",
            "Permission / Time Check": "Asking if the target has time or permission to continue.",
            "Charity Awareness Probe": "Asking if the target knows about the charity or cause.",
            "Qualification / Segmentation": "Asking questions to learn about the target's background or views.",
            "Donation Baseline / Habit Probe": "Asking about the target's prior donation habits.",
            "Logistics / Coordination": "Discussing logistics of the donation process.",
            "Acknowledgement": "Acknowledging, thanking, or affirming the target's response.",
            "Conversation Closing": "Ending the conversation, saying goodbye.",
            "Non-persuasive Other": "Other non-persuasive turns that don't fit above categories."
        }
    }
}

# ============================================================
# BUILD TAXONOMY STRING FOR PROMPT
# ============================================================

def build_taxonomy_string():
    lines = []
    for cat_name, cat_data in TAXONOMY.items():
        lines.append(f"\n## {cat_name}")
        lines.append(f"Description: {cat_data['description']}")
        lines.append("Strategies:")
        for strat_name, strat_desc in cat_data["strategies"].items():
            lines.append(f"  - {strat_name}: {strat_desc}")
    return "\n".join(lines)

TAXONOMY_STRING = build_taxonomy_string()

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a hierarchical persuasion strategy classification system. Your task is to classify persuader utterances from charitable donation dialogues into a two-level taxonomy.

STEP 1: Select the most appropriate CATEGORY from the 12 categories below.
STEP 2: Select the most appropriate STRATEGY within that category.

Return your answer as JSON with exactly two fields:
{"category": "<category name>", "strategy": "<strategy name>"}

Use ONLY the exact category and strategy names listed below. Do not invent new ones.

TAXONOMY:
""" + TAXONOMY_STRING + """

DECISION RULES:
- If the utterance is a greeting, acknowledgement, or non-persuasive, use "Conversation Management".
- If multiple strategies apply, choose the PRIMARY one (the dominant persuasive intent).
- If the utterance contains a direct request to donate, consider "Call to Action" unless another strategy is more dominant.
- Classify based on the persuader's INTENT, not just the surface words.
"""

# ============================================================
# ANNOTATION FUNCTION
# ============================================================

def build_context(dialog_df, current_idx, n_context=CONTEXT_TURNS):
    """Build context string from up to n previous turns."""
    rows_before = dialog_df[dialog_df.index < current_idx].tail(n_context)
    context_lines = []
    for _, row in rows_before.iterrows():
        role = "PERSUADER" if row["B4"] == 0 else "TARGET"
        context_lines.append(f"[{role}, Turn {row['Turn']}]: {row['Unit']}")
    return "\n".join(context_lines)


def annotate_turn(utterance, context, session, retries=3):
    """Send one utterance to Qwen3:30b and parse the response."""

    user_msg = ""
    if context:
        user_msg += f"DIALOGUE CONTEXT (previous turns):\n{context}\n\n"
    user_msg += f"UTTERANCE TO CLASSIFY:\n{utterance}\n\n"
    user_msg += 'Respond with JSON only: {"category": "...", "strategy": "..."}'

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE
        }
    }

    for attempt in range(retries + 1):
        try:
            resp = session.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]

            # Try to extract JSON from the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                cat = result.get("category", "")
                strat = result.get("strategy", "")
                if cat and strat:
                    return cat, strat
                # Got JSON but empty fields — retry
                if attempt < retries:
                    time.sleep(2)
                    continue
                return cat, strat
            else:
                if attempt < retries:
                    time.sleep(2)
                    continue
                return "", ""

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < retries:
                time.sleep(5)
                continue
            return "", ""

    return "", ""


# ============================================================
# MAIN
# ============================================================

def main():
    # Setup logging
    log_f = open(LOG_FILE, "a")
    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"Starting Qwen3:30b annotation")
    log(f"Model: {MODEL}, Server: {OLLAMA_URL}")
    log(f"Temperature: {TEMPERATURE}, Context turns: {CONTEXT_TURNS}")

    # Load data
    df = pd.read_csv(INPUT_CSV)
    log(f"Loaded {len(df)} rows, {df['B2'].nunique()} dialogues")

    # Get persuader turns only
    persuader = df[df["B4"] == 0].copy()
    log(f"Persuader turns to annotate: {len(persuader)}")

    # Check for checkpoint
    start_idx = 0
    results = []
    if os.path.exists(CHECKPOINT_CSV):
        checkpoint = pd.read_csv(CHECKPOINT_CSV)
        start_idx = len(checkpoint)
        results = checkpoint.to_dict("records")
        log(f"Resuming from checkpoint: {start_idx} already done")

    # Create session for connection pooling
    session = requests.Session()

    # First, trigger model load
    log("Loading model (first request may be slow)...")
    test_cat, test_strat = annotate_turn("Hello, how are you?", "", session)
    log(f"Model loaded. Test: category='{test_cat}', strategy='{test_strat}'")

    # Group by dialogue for context building
    dialogues = df.groupby("B2")
    persuader_indices = persuader.index.tolist()

    total = len(persuader_indices)
    t_start = time.time()
    errors = 0

    for i, idx in enumerate(persuader_indices):
        if i < start_idx:
            continue

        row = df.loc[idx]
        dialog_id = row["B2"]
        dialog_df = dialogues.get_group(dialog_id)

        utterance = str(row["Unit"])
        context = build_context(dialog_df, idx)

        cat, strat = annotate_turn(utterance, context, session)

        if not strat:
            errors += 1

        results.append({
            "original_index": idx,
            "dialog_id": dialog_id,
            "turn": row["Turn"],
            "utterance": utterance,
            "qwen_category": cat,
            "qwen_strategy": strat
        })

        # Progress & checkpoint
        done = i + 1
        if done % 50 == 0:
            elapsed = time.time() - t_start
            rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            eta_min = (total - done) / rate / 60 if rate > 0 else 0

            # Save checkpoint
            pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)

            log(f"Progress: {done}/{total} ({done*100/total:.1f}%) | "
                f"Rate: {rate:.1f} turns/s | ETA: {eta_min:.0f} min | "
                f"Errors: {errors}")

    # Save final results
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    elapsed_total = (time.time() - t_start) / 60
    valid = result_df["qwen_strategy"].ne("").sum()
    log(f"DONE. Total: {total}, Valid: {valid} ({valid*100/total:.1f}%), "
        f"Errors: {errors}, Time: {elapsed_total:.1f} min")
    log(f"Results saved to {OUTPUT_CSV}")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)

    log_f.close()


if __name__ == "__main__":
    main()
