"""
prompt_builder.py - Footing project

Builds an extraction prompt dynamically from a shared base template plus a
per-pattern rules file, instead of keeping 10 fully duplicated prompt files.

Layout
------
    prompts/
        _base.txt              shared template with $placeholders
        rules/
            pattern_1.txt      ONLY the per-pattern rules block
            ...
            pattern_10.txt

The base template exposes three placeholders:
    $role          - the "You are an expert ..." intro line
    $json_schema   - the target JSON shape (flat or stepped)
    $pattern_rules - the pattern-specific rules block

Most patterns share the FLAT size schema. Patterns 6 and 7 use a STEPPED
size schema (step_1 / step_2), so the correct schema is selected per pattern.
"""

import os
from string import Template


PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
RULES_DIR = os.path.join(PROMPTS_DIR, "rules")
BASE_PATH = os.path.join(PROMPTS_DIR, "_base.txt")


# Two known size-schema variants used across the 10 footing patterns.
FLAT_SCHEMA = """{
  "footings": [
    {
      "footing_id": "",
      "column_id": null,
      "size": {
        "width": null,
        "depth": null,
        "length": null
      },
      "reinforcement": {
        "dia": [],
        "spacing": []
      },
      "nos": null,
      "mix": null,
      "steel_grade": null
    }
  ]
}"""

STEPPED_SCHEMA = """{
  "footings": [
    {
      "footing_id": "",
      "column_id": null,
      "size": {
        "step_1": {
          "width": null,
          "depth": null,
          "length": null
        },
        "step_2": {
          "width": null,
          "depth": null,
          "length": null
        }
      },
      "reinforcement": {
        "dia": [],
        "spacing": []
      },
      "nos": null,
      "mix": null,
      "steel_grade": null
    }
  ]
}"""


DEFAULT_ROLE = (
    "You are an expert structural drawing extractor specialized in RCC footing tables."
)


# Per-pattern configuration: the intro role line and which size schema to use.
# Rules text lives in prompts/rules/pattern_<N>.txt.
PATTERN_CONFIG = {
    1: {
        "role": "You are an expert structural drawing extractor specialized in RCC footing tables.",
        "schema": FLAT_SCHEMA,
    },
    2: {
        "role": "You are an expert RCC footing schedule reader.",
        "schema": FLAT_SCHEMA,
    },
    3: {
        "role": "You are an expert structural drawing extractor specialized in RCC footing schedules.",
        "schema": FLAT_SCHEMA,
    },
    4: {
        "role": "You are an expert structural drawing extractor specialized in RCC footing schedules.",
        "schema": FLAT_SCHEMA,
    },
    5: {
        "role": "You are an expert structural drawing extractor specialized in RCC footing detail tables.",
        "schema": FLAT_SCHEMA,
    },
    6: {
        "role": "You are an expert structural drawing extractor specialized in RCC footing schedules.",
        "schema": STEPPED_SCHEMA,
    },
    7: {
        "role": (
            "You are an expert structural RCC footing schedule extractor.\n\n"
            "You are extracting from a structured grid table.\n\n"
            "You MUST respect exact row-column alignment."
        ),
        "schema": STEPPED_SCHEMA,
    },
    8: {
        "role": "You are extracting a structured FOOTING SCHEDULE table.\n\nThe table contains labeled rows.",
        "schema": FLAT_SCHEMA,
    },
    9: {
        "role": (
            "You are extracting a structured FOOTING SCHEDULE table.\n\n"
            "This table contains RAFT columns:\n"
            "RAFT-1, RAFT-2, RAFT-3, etc."
        ),
        "schema": FLAT_SCHEMA,
    },
    10: {
        "role": "You are an expert structural drawing extractor specialized in RCC footing detail tables.",
        "schema": FLAT_SCHEMA,
    },
}


def _read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _rules_path(pattern_number):
    return os.path.join(RULES_DIR, f"pattern_{pattern_number}.txt")


def build_prompt(pattern_number):
    """
    Assemble the full extraction prompt for a footing pattern.

    Raises FileNotFoundError if the rules file for the pattern is missing,
    and ValueError if the pattern number is unknown.
    """

    pattern_number = int(pattern_number)
    config = PATTERN_CONFIG.get(pattern_number)
    if config is None:
        raise ValueError(f"Unknown footing pattern: {pattern_number}")

    rules_path = _rules_path(pattern_number)
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Missing rules file for pattern {pattern_number}: {rules_path}")

    template = Template(_read(BASE_PATH))
    return template.substitute(
        role=config.get("role", DEFAULT_ROLE),
        json_schema=config["schema"],
        pattern_rules=_read(rules_path).strip(),
    )


if __name__ == "__main__":
    # Smoke test: build every configured pattern prompt.
    for n in sorted(PATTERN_CONFIG):
        prompt = build_prompt(n)
        print(f"--- pattern {n}: {len(prompt)} chars ---")
