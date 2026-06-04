import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools
from prompt_builder import build_prompt


# ===============================
# LOAD PROMPT
# ===============================

def load_prompt():
    return build_prompt(3)


# ===============================
# EXPAND COLUMN GROUPS
# ===============================

def expand_columns(text):
    if not text:
        return None

    text = text.replace("\n", " ").strip()
    tokens = text.split()
    result = []
    current_prefix = ""

    for token in tokens:
        token = token.strip(",")
        first_digit_idx = None
        for i, ch in enumerate(token):
            if ch.isdigit():
                first_digit_idx = i
                break
        if first_digit_idx is None:
            continue
        if first_digit_idx > 0:
            current_prefix = token[:first_digit_idx]
            numbers_part = token[first_digit_idx:]
        else:
            numbers_part = token
        nums = [n.strip() for n in numbers_part.split(",") if n.strip()]
        for n in nums:
            result.append(f"{current_prefix}{n}")

    return ",".join(result) if result else None


# ===============================
# NORMALISE TO LIST
# ===============================

def _to_list(value):
    if isinstance(value, list):
        return sorted(set(str(v).strip() for v in value if str(v).strip()))
    if value:
        return [str(value).strip()]
    return []


# ===============================
# PASS 2 — MIX EXTRACTION
# ===============================

MIX_PROMPT = """You are reading a structural footing schedule table image.

TASK: Find EVERY piece of text that matches this pattern: the letter M followed by 2 or more digits.
Examples of valid values: M15, M20, M25, M30, M35, M40

WHERE TO LOOK — check ALL of these:
1. Vertical (rotated) text labels on the LEFT side of the table.
2. Combined labels like "M25 : Fe500" — the M25 part is a match. INCLUDE IT.
3. Standalone labels like "M15" — include it.
4. Any row header or cell that contains an M followed by digits.

DO NOT skip M25 just because it appears together with Fe500 in the same label.
The pattern "M25 : Fe500" contains TWO separate values: M25 (concrete grade) and Fe500 (steel grade).
You must extract M25.

Return ONLY this JSON — no explanation, no extra text:
{"mix": ["M25", "M15"]}"""


def _scrape_m_grades(text):
    """Extract all M-grade patterns (M15, M25, etc.) from any raw text string."""
    print(text[:120])
    return re.findall(r'\bM-?\s*\d{2,3}\b', str(text), re.IGNORECASE)


def extract_mix(img_path):
    print("  [pass-mix] extracting concrete mix grades...")
    raw = extract_from_image(img_path, MIX_PROMPT)
    try:
        raw_text = raw if isinstance(raw, str) else json.dumps(raw)
        if isinstance(raw, str):
            raw = json.loads(raw)
            print(f"This is the raw response From Mix Extraction Prompt: {raw}")
        values = raw.get("mix", [])
        # Also scrape any M-grades the model wrote anywhere in its response
        scraped = _scrape_m_grades(raw_text)
        all_values = list(values) + scraped
        result = sorted(set(str(v).strip().upper() for v in all_values if str(v).strip()))
        # Normalise: "M 25" or "M-25" → "M25"
        result = sorted(set(re.sub(r'M[-\s]*(\d+)', r'M\1', v) for v in result))
        print(f"  [pass-mix] found: {result}")
        return result
    except Exception as e:
        print(f"  [pass-mix] parse failed: {e} | raw={str(raw)[:120]}")
        return []


# ===============================
# PASS 3 — STEEL GRADE EXTRACTION
# ===============================

STEEL_PROMPT = """You are reading a structural footing schedule table image.

Your ONLY task: find ALL steel grade values written ANYWHERE in the table.

Rules:
- A vertical label like "M25 : Fe500" contains Fe500 (steel grade). Extract Fe500.
- Steel grades match the pattern Fe followed by a number: Fe500, Fe450, Fe415, etc.
- Look at every vertical label in the left-side header columns of the table.

Return ONLY this JSON — nothing else:
{"steel_grade": ["Fe500"]}

No explanation. No extra text."""


def extract_steel_grade(img_path):
    print("  [pass-steel] extracting steel grades...")
    raw = extract_from_image(img_path, STEEL_PROMPT)
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        values = raw.get("steel_grade", [])
        result = sorted(set(str(v).strip() for v in values if str(v).strip()))
        print(f"  [pass-steel] found: {result}")
        return result
    except Exception as e:
        print(f"  [pass-steel] parse failed: {e}")
        return []


# ===============================
# PASS 4 — REINFORCEMENT EXTRACTION
# ===============================

REINF_PROMPT = """You are reading a structural footing schedule table image.

Your ONLY task: find ALL reinforcement bar specifications from the BOTTOM STEEL and TOP STEEL rows.

Rules:
- Look only at SHORT DIR and LONG DIR cells under TOP STEEL and BOTTOM STEEL.
- Format is like: T10@150 C/C, T12@100 C/C
  - Dia = T10 or T12
  - Spacing = 150 C/C or 100 C/C
- Ignore "---" or empty cells.
- Collect ALL unique dia and spacing values across the entire table.

Return ONLY this JSON — nothing else:
{"dia": ["T10", "T12"], "spacing": ["100 C/C", "150 C/C"]}

No explanation. No extra text."""


def extract_reinforcement(img_path):
    print("  [pass-reinf] extracting reinforcement...")
    raw = extract_from_image(img_path, REINF_PROMPT)
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        dia = sorted(set(str(v).strip() for v in raw.get("dia", []) if str(v).strip()))
        spacing = sorted(set(str(v).strip() for v in raw.get("spacing", []) if str(v).strip()))
        print(f"  [pass-reinf] dia={dia} spacing={spacing}")
        return {"dia": dia, "spacing": spacing}
    except Exception as e:
        print(f"  [pass-reinf] parse failed: {e}")
        return {"dia": [], "spacing": []}


# ===============================
# PROCESS PDF
# ===============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    structure_prompt = load_prompt()
    final_footings = []
    footing_counter = 1

    for img_path in tqdm(image_paths):

        print(f"\n── Page: {os.path.basename(img_path)} ──")

        # Pass 1: structure — column IDs, size, depth (tools-based)
        print("  [pass-1] extracting structure (column IDs, size, depth)...")
        result = extract_with_tools(img_path, structure_prompt)

        if isinstance(result, dict):
            parsed = result
        else:
            try:
                parsed = json.loads(result)
            except Exception:
                print("  ⚠ Pass 1 JSON parse failed, skipping page")
                continue

        # Pass 2: mix values (dedicated focused prompt)
        page_mix = extract_mix(img_path)

        # Pass 3: steel grades (dedicated focused prompt)
        page_steel = extract_steel_grade(img_path)

        # Pass 4: reinforcement (dedicated focused prompt)
        page_reinf = extract_reinforcement(img_path)

        for footing in parsed.get("footings", []):

            raw_column = footing.get("column_id", "")
            if isinstance(raw_column, dict):
                raw_column = " ".join(str(v) for v in raw_column.values())
            else:
                raw_column = str(raw_column)
            expanded_columns = expand_columns(raw_column)

            # Use per-footing reinforcement from pass 1; fall back to page-level pass 4
            f_reinf = footing.get("reinforcement") or {}
            dia = _to_list(f_reinf.get("dia")) or page_reinf["dia"]
            spacing = _to_list(f_reinf.get("spacing")) or page_reinf["spacing"]

            # Mix and steel: prefer page-level dedicated pass over pass-1 values
            mix = page_mix or _to_list(footing.get("mix"))
            steel_grade = page_steel or _to_list(footing.get("steel_grade"))

            record = {
                "footing_id": str(footing_counter),
                "column_id": expanded_columns,
                "size": footing.get("size"),
                "reinforcement": {"dia": dia, "spacing": spacing},
                "nos": footing.get("nos"),
                "mix": mix,
                "steel_grade": steel_grade,
            }
            final_footings.append(record)

            size = footing.get("size") or {}
            print(
                f"  [footing {footing_counter:>3}] col={expanded_columns} | "
                f"size={size.get('width')}x{size.get('length')}x{size.get('depth')} | "
                f"dia={dia} spacing={spacing} | mix={mix} steel={steel_grade}"
            )
            footing_counter += 1

    output_data = {"footings": final_footings}
    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  ── Extraction summary: {file_name} ──")
    print(f"  Total footings : {len(final_footings)}")
    all_mixes = sorted(set(m for f in final_footings for m in (f.get("mix") or [])))
    all_grades = sorted(set(g for f in final_footings for g in (f.get("steel_grade") or [])))
    print(f"  Mix values     : {all_mixes}")
    print(f"  Steel grades   : {all_grades}")
    print(f"  Output file    : {output_file}")
    print(f"  ─────────────────────────────────────────")
    print(f"\n✅ Output saved to {output_file}")


# ===============================
# MAIN
# ===============================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
