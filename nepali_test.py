"""
Romanized Nepali robustness test for local Ollama models.

Usage:
    pip install requests
    ollama serve            # if not already running
    python nepali_test.py

Outputs:
    comprehension_results.csv   (every response + similarity score)
    generation_results.csv      (replies for you to rate manually, column 'rating')
Prints a summary table: average score per model per variation type.
"""
import csv
import math
from collections import defaultdict

import requests

OLLAMA = "http://localhost:11434"
MODELS = ["qwen3:8b", "llama3.2:latest", "qwen3.5:2b"]
EMBED_MODEL = "bge-m3:latest"

# ---------------------------------------------------------------------------
# DATASET: (reference English, [(variation_type, romanized text), ...])
# The first variant of each sentence should be the "baseline" spelling.
# Add more sentences and variants freely.
# ---------------------------------------------------------------------------
DATA = [
    ("I am going home.", [
        ("baseline", "ma ghar janchu"),
        ("ch_to_x", "mah ghar janxu"),
        ("chh_double_vowel", "ma ghar janchhuu"),
        ("dropped_vowel", "m ghar jancu"),
        ("long_vowel", "ma ghar jaanchu"),
    ]),
    ("What is your name?", [
        ("baseline", "timro naam ke ho"),
        ("short_slang", "timro nam k ho"),
        ("dropped_vowel", "tmro naam ke ho"),
        ("vowel_sub", "timro naam kae ho"),
    ]),
    ("I am hungry.", [
        ("baseline", "malai bhok lagyo"),
        ("h_added", "malai bhokh lagyo"),
        ("dropped_vowel", "mlai bhok lagyo"),
        ("double_vowel", "malai bhok lagyoo"),
    ]),
    ("Where are you going?", [
        ("baseline", "timi kaha janchau"),
        ("ch_to_x", "timi kaha janxau"),
        ("dropped_vowel", "tmi kaha jaanchau"),
        ("chh_long_vowel", "timi kahaa janchhau"),
    ]),
    ("I like eating rice.", [
        ("baseline", "malai bhat khana man parchha"),
        ("ch_to_x", "malai bhat khana man parxa"),
        ("dropped_vowel", "mlai bhaat khana man parcha"),
        ("double_n", "malai bhat khana mann parchha"),
    ]),
    ("It is raining today.", [
        ("baseline", "aaja paani parirako chha"),
        ("ch_to_x", "aaja pani pariraxa"),
        ("short_a", "aja pani pariracha"),
        ("x_ending", "aaja paani parirako xa"),
    ]),
]

# Generation test: Romanized Nepali questions (rate replies manually afterwards)
GEN_PROMPTS = [
    "timro naam ke ho?",
    "tmro nam k ho",
    "kasto cha aaja?",
    "kasto xa aja",
    "tapai kaha basnu hunchha?",
    "malai ek ota kabita sunau na",
]

TRANSLATE_PROMPT = (
    "The following sentence is Nepali written in Latin (Roman) letters, "
    "with informal spelling. Translate it into English. "
    "Reply with ONLY the English translation, nothing else.\n\n{text}"
)
GEN_SYSTEM = (
    "You are a friendly assistant. Reply ONLY in Romanized Nepali "
    "(Nepali written in English letters, not Devanagari). Keep it short."
)


def chat(model, messages):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0}}
    # Turn off "thinking" for models that support it (qwen3 family)
    body["think"] = False
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=300)
    if r.status_code != 200:  # model may not support the think flag
        body.pop("think")
        r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def run_comprehension():
    rows = []
    for model in MODELS:
        print(f"\n== Comprehension: {model}")
        for ref, variants in DATA:
            for vtype, text in variants:
                out = chat(model, [{"role": "user",
                                    "content": TRANSLATE_PROMPT.format(text=text)}])
                ref_emb, out_emb = embed([ref, out])
                score = cosine(ref_emb, out_emb)
                rows.append({"model": model, "reference": ref,
                             "variation_type": vtype, "romanized": text,
                             "model_output": out, "similarity": round(score, 4)})
                print(f"  {score:.2f} | {text!r} -> {out[:60]!r}")
    with open("comprehension_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def run_generation():
    rows = []
    for model in MODELS:
        print(f"\n== Generation: {model}")
        for prompt in GEN_PROMPTS:
            out = chat(model, [{"role": "system", "content": GEN_SYSTEM},
                               {"role": "user", "content": prompt}])
            rows.append({"model": model, "prompt": prompt,
                         "reply": out.replace("\n", " "),
                         "rating": ""})  # fill in by hand: 0 / 1 / 2
            print(f"  {prompt!r} -> {out[:70]!r}")
    with open("generation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    by_model = defaultdict(list)
    by_type = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_model[r["model"]].append(r["similarity"])
        by_type[r["model"]][r["variation_type"]].append(r["similarity"])
    print("\n=== SUMMARY (mean cosine similarity to reference; higher = better)")
    for model, scores in by_model.items():
        print(f"\n{model}: overall {sum(scores)/len(scores):.3f}")
        base = by_type[model].get("baseline", [])
        base_avg = sum(base) / len(base) if base else None
        for vtype, s in sorted(by_type[model].items()):
            avg = sum(s) / len(s)
            drop = f"  (vs baseline {avg - base_avg:+.3f})" if base_avg is not None and vtype != "baseline" else ""
            print(f"   {vtype:18s} {avg:.3f}{drop}")


if __name__ == "__main__":
    summarize(run_comprehension())
    run_generation()
    print("\nDone. Open generation_results.csv and fill the 'rating' column.")