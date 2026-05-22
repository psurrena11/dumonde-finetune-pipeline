"""
Rebuild interviews-sft.jsonl with taxonomy-aware system prompts.

Reads demographics from the screener CSV and optionally fetches intersection
scaffolding from duMonde. Matches interviews to screener rows sequentially
(interview 0 = CSV row 0) or via a custom --mapping JSON file.

Usage:
  python prepare_training.py \\
    --jsonl interviews-sft.jsonl \\
    --csv interviews.csv \\
    --out interviews-sft-v2.jsonl

  python prepare_training.py \\
    --jsonl interviews-sft.jsonl \\
    --csv interviews.csv \\
    --out interviews-sft-v2.jsonl \\
    --dumonde-url https://api.dumonde.ai \\
    --admin-secret YOUR_SECRET \\
    --mapping mapping.json

mapping.json format (optional, overrides sequential):
  [{"interview_index": 0, "csv_index": 5}, {"interview_index": 1, "csv_index": 2}, ...]

For interviews with no CSV match, the original generic system prompt is kept.
"""

import argparse
import csv
import json
import re
import sys
import urllib.request
import urllib.error

Q1_KEYWORD = "Can you start by describing yourself"
SYSTEM_PROMPT_GENERIC = (
    "You are a person being interviewed about your political worldview. "
    "Answer in your own authentic voice: specific, grounded in your actual "
    "experience, and honest about where you contradict yourself. Do not sound "
    "like a policy document or a survey response."
)


def clean_region(raw):
    if not raw:
        return None
    # Strip parentheticals first, then split on delimiters
    no_parens = re.sub(r"\s*\([^)]*\)", "", raw)
    return no_parens.split(";")[0].split(",")[0].strip()


def clean_leaning(raw):
    if not raw:
        return None
    mapping = {
        "very liberal": "far-left",
        "liberal": "lean-left",
        "moderate / centrist": "center",
        "moderate/centrist": "center",
        "conservative": "lean-right",
        "very conservative": "far-right",
    }
    return mapping.get(raw.strip().lower(), raw.strip().lower())


def clean_generation(raw):
    if not raw:
        return None
    return re.sub(r"\s*\(.*?\)", "", raw).strip()


def clean_socioeconomic(raw):
    if not raw:
        return None
    # Take everything before the first parenthesis
    return re.sub(r"\s*\(.*?\)", "", raw).strip()


def build_persona_description(row):
    """Build a short natural-language identity string from CSV row."""
    leaning_raw   = row.get("Part 2: Taxonomy - Political Leaning (rough sense)", "")
    race          = row.get("Part 2: Taxonomy - Race / Ethnicity (Check all that apply)", "")
    religion      = row.get("Part 2: Taxonomy - Religion / Spirituality (Check all that apply)", "")
    generation    = row.get("Part 2: Taxonomy - Generation / Age", "")
    region        = row.get("Part 2: Taxonomy - Region / Geography (Check all that apply)", "")
    ses           = row.get(
        "Part 2: Taxonomy - Socioeconomic Background (Check the one that best fits your upbringing, not your current status)", ""
    )
    education     = row.get("Part 2: Taxonomy - Education Level (Check highest attained)", "")
    cultural_bg   = row.get(
        "Part 3: Your Cultural Frame (Open-Ended) - How would you describe your cultural background in your own words? (2-3 sentences is fine)", ""
    )

    leaning_label = {
        "very liberal": "very liberal",
        "liberal": "liberal",
        "moderate / centrist": "centrist",
        "moderate/centrist": "centrist",
        "conservative": "conservative",
        "very conservative": "very conservative",
    }.get(leaning_raw.strip().lower(), leaning_raw.strip())

    region_clean  = clean_region(region)
    gen_clean     = clean_generation(generation)
    ses_clean     = clean_socioeconomic(ses)

    parts = []
    if leaning_label:
        parts.append(leaning_label)
    if race:
        parts.append(race.split(";")[0].strip())
    if religion:
        parts.append(religion.split(";")[0].strip())
    if gen_clean:
        parts.append(gen_clean)
    if region_clean:
        parts.append(f"from the {region_clean}")
    if ses_clean:
        parts.append(f"{ses_clean} background")
    if education:
        parts.append(f"{education} education")

    identity = ", ".join(parts) if parts else "person with a specific cultural perspective"
    return identity, cultural_bg.strip()


def build_system_prompt(row, scaffolding=None):
    identity, cultural_bg = build_persona_description(row)

    prompt = (
        f"You are a {identity}. "
        "Answer in your own authentic voice: specific, grounded in your actual experience, "
        "and honest about where you contradict yourself. "
        "Do not sound like a policy document or a survey response."
    )

    if cultural_bg:
        prompt += f"\n\nIn their own words, this person describes their background as: \"{cultural_bg}\""

    if scaffolding:
        prompt += f"\n\nCultural context for this perspective:\n{scaffolding}"

    return prompt


def build_background_prefix(row):
    leaning_raw = row.get("Part 2: Taxonomy - Political Leaning (rough sense)", "")
    race        = row.get("Part 2: Taxonomy - Race / Ethnicity (Check all that apply)", "").split(";")[0].strip()
    religion    = row.get("Part 2: Taxonomy - Religion / Spirituality (Check all that apply)", "").split(";")[0].strip()
    generation  = clean_generation(row.get("Part 2: Taxonomy - Generation / Age", ""))
    region      = clean_region(row.get("Part 2: Taxonomy - Region / Geography (Check all that apply)", ""))

    parts = [p for p in [leaning_raw.strip(), race, religion, generation, region] if p]
    return "Background: " + ", ".join(parts) + "." if parts else "Background: not specified."


def fetch_scaffolding(dumonde_url, admin_secret, country, leaning):
    """Fetch intersection scaffolding from duMonde admin API."""
    if not dumonde_url or not admin_secret:
        return None
    try:
        url = f"{dumonde_url.rstrip('/')}/v1/admin/spectrum?country={country}"
        req = urllib.request.Request(url, headers={"X-Admin-Secret": admin_secret})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            positions = data.get("data", data)
            if isinstance(positions, list):
                for p in positions:
                    if p.get("leaning_key") == leaning:
                        return p.get("scaffolding") or None
    except Exception as e:
        print(f"  Warning: scaffolding fetch failed ({e})", file=sys.stderr)
    return None


def split_into_interviews(examples):
    """Split flat list of examples into per-interview groups using Q1 as boundary."""
    interviews = []
    current = []
    for ex in examples:
        user_content = ex["messages"][1]["content"] if len(ex["messages"]) > 1 else ""
        if Q1_KEYWORD in user_content and current:
            interviews.append(current)
            current = []
        current.append(ex)
    if current:
        interviews.append(current)
    return interviews


def main():
    parser = argparse.ArgumentParser(description="Rebuild training JSONL with taxonomy system prompts.")
    parser.add_argument("--jsonl",         required=True, help="Input JSONL path")
    parser.add_argument("--csv",           required=True, help="Screener CSV path")
    parser.add_argument("--out",           required=True, help="Output JSONL path")
    parser.add_argument("--mapping",       help="Optional JSON mapping file [{interview_index, csv_index}, ...]")
    parser.add_argument("--dumonde-url",   help="duMonde API URL for scaffolding fetch")
    parser.add_argument("--admin-secret",  help="duMonde admin secret")
    parser.add_argument("--country",       default="us", help="Country code for scaffolding lookup (default: us)")
    args = parser.parse_args()

    # Load examples
    examples = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Load CSV
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))

    # Split into per-interview groups
    interviews = split_into_interviews(examples)
    print(f"Found {len(interviews)} interviews, {len(csv_rows)} CSV rows", file=sys.stderr)

    # Load custom mapping if provided
    custom_mapping = {}
    if args.mapping:
        with open(args.mapping) as f:
            for entry in json.load(f):
                custom_mapping[entry["interview_index"]] = entry["csv_index"]

    # Scaffolding cache to avoid repeated API calls
    scaffolding_cache = {}

    out_examples = []

    for i, interview in enumerate(interviews):
        csv_index = custom_mapping.get(i, i)  # default sequential

        if csv_index < len(csv_rows):
            row = csv_rows[csv_index]
            name = row.get("Part 1: Contact Information (Required) - Full Name", f"Interview {i+1}").strip()
            leaning_key = clean_leaning(row.get("Part 2: Taxonomy - Political Leaning (rough sense)", ""))

            # Fetch scaffolding (cached per leaning)
            scaffolding = None
            if args.dumonde_url and leaning_key:
                cache_key = f"{args.country}:{leaning_key}"
                if cache_key not in scaffolding_cache:
                    scaffolding_cache[cache_key] = fetch_scaffolding(
                        args.dumonde_url, args.admin_secret, args.country, leaning_key
                    )
                scaffolding = scaffolding_cache[cache_key]

            system_prompt   = build_system_prompt(row, scaffolding)
            background_line = build_background_prefix(row)
            print(f"  [{i+1}/{len(interviews)}] {name} (CSV row {csv_index})", file=sys.stderr)
        else:
            system_prompt   = SYSTEM_PROMPT_GENERIC
            background_line = None
            print(f"  [{i+1}/{len(interviews)}] No CSV match — keeping generic prompt", file=sys.stderr)

        for ex in interview:
            msgs = list(ex["messages"])

            # Replace system prompt
            if msgs[0]["role"] == "system":
                msgs[0] = {"role": "system", "content": system_prompt}

            # Replace "Background: undefined." in user message
            if background_line and len(msgs) > 1 and msgs[1]["role"] == "user":
                msgs[1] = {
                    "role": "user",
                    "content": re.sub(
                        r"^Background:.*?\n\n",
                        background_line + "\n\n",
                        msgs[1]["content"],
                        flags=re.DOTALL,
                    ),
                }

            out_examples.append({"messages": msgs})

    with open(args.out, "w") as f:
        for ex in out_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(out_examples)} examples to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
