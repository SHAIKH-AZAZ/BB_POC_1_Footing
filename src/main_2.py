import os
import re
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools
from prompt_builder import build_prompt


# ===============================
# LOAD PROMPT
# ===============================
def load_prompt():
    return build_prompt(2)


# ===============================
# POST-PROCESS GUARDS (pattern 2)
# ===============================
_SIZE_RE = re.compile(r"\d{3,4}\s*[xX]\s*\d{3,4}")
_REFER_RE = re.compile(r"REFER|DETAIL\s+SECTION", re.IGNORECASE)
# Footing reinforcement in this pattern is ALWAYS count-dia TOR, e.g. "25-10T".
_COMPOUND_DIA_RE = re.compile(r"^\d{1,3}-\d{1,2}T$")


def _null_footing(footing):
    footing["size"] = {"width": None, "depth": None, "length": None}
    footing["reinforcement"] = {"dia": [], "spacing": []}
    footing["mix"] = None
    return footing


def apply_guards(footing):
    """
    Deterministic correctness guards for pattern 2:

    1. Refer-only columns -> null. If the raw footing cell text contains a
       REFER/DETAIL note and has NO numeric footing size (NNNN X NNNN), the
       column has no footing data even if the model guessed values from a
       neighbouring column.
    2. Reinforcement must be the count-dia TOR form (e.g. "25-10T"). Plain
       values like "12T"/"10T" come from the column ring rows, not the footing
       REINF //B //L rows, so drop them.
    """
    cell = footing.get("footing_cell_text") or ""

    # Guard 1: refer-only -> null
    if cell and _REFER_RE.search(cell) and not _SIZE_RE.search(cell):
        _null_footing(footing)
        footing.pop("footing_cell_text", None)
        return footing

    # Guard 2: keep only compound count-dia TOR reinforcement
    reinf = footing.get("reinforcement", {}) or {}
    dia = [d for d in reinf.get("dia", []) if _COMPOUND_DIA_RE.match(str(d))]
    footing["reinforcement"] = {"dia": dia, "spacing": []}

    footing.pop("footing_cell_text", None)
    return footing


# ===============================
# PROCESS PDF
# ===============================
def process_pdf(pdf_path):

    file_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    file_output_folder = os.path.join(
        OUTPUT_DIR,
        file_name
    )

    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")

    image_paths = convert_pdf_to_images(
        pdf_path,
        file_output_folder
    )

    if not image_paths:
        raise Exception("No images generated from PDF.")

    prompt = load_prompt()

    final_footings = []
    footing_counter = 1

    # ===============================
    # RUN VISION EXTRACTION
    # ===============================
    for img_path in tqdm(image_paths):

        result = extract_with_tools(img_path, prompt, enforce_zoom=True)

        # Model sometimes returns dict already
        if isinstance(result, dict):
            parsed = result
        else:
            try:
                parsed = json.loads(result)
            except Exception:
                print("⚠ JSON parsing failed")
                print(result)
                continue

        footings = parsed.get("footings", [])

        if not footings:
            continue

        # Apply deterministic correctness guards, then assign sequential IDs
        for footing in footings:

            footing = apply_guards(footing)

            footing["footing_id"] = str(footing_counter)
            footing_counter += 1

            final_footings.append(footing)

    # ===============================
    # SAVE OUTPUT
    # ===============================
    output_data = {
        "footings": final_footings
    }

    output_file = os.path.join(
        file_output_folder,
        f"{file_name}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Output saved to {output_file}")


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
        print("⚠ No PDF files found.")
        return

    for pdf in pdf_files:
        process_pdf(
            os.path.join(INPUT_DIR, pdf)
        )


if __name__ == "__main__":
    main()
