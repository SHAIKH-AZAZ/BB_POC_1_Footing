import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


# ===========================
# Load Prompt
# ===========================
def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_6.txt"), "r") as f:
        return f.read()


# ===========================
# Parse L x B correctly
# ===========================
def parse_lxb(value):
    if not value or value == "-":
        return None, None

    nums = re.findall(r"\d+", str(value))
    if len(nums) >= 2:
        length = int(nums[0])   # FIRST = LENGTH
        width = int(nums[1])    # SECOND = WIDTH
        return length, width

    return None, None


# ===========================
# Normalize Reinforcement
# ===========================
def normalize_reinforcement(dia_list, spacing_list):

    clean_dia = []
    clean_spacing = []

    for d in dia_list:
        if not d:
            continue
        d = d.upper().replace("TOR", "T")
        d = re.sub(r"\s+", "", d)
        if d not in clean_dia:
            clean_dia.append(d)

    for s in spacing_list:
        if not s:
            continue
        s = s.upper().strip()
        numbers = re.findall(r"\d+", s)
        if numbers:
            s = f"{numbers[0]} C/C"
        if s not in clean_spacing:
            clean_spacing.append(s)

    return clean_dia, clean_spacing


# ===========================
# Clean Footing
# ===========================
def clean_footing(footing):

    size = footing.get("size") or {}
    step1 = size.get("step_1") or {}
    step2 = size.get("step_2") or {}

    reinf = footing.get("reinforcement") or {}

    dia, spacing = normalize_reinforcement(
        reinf.get("dia", []),
        reinf.get("spacing", [])
    )

    return {
        "footing_id": footing.get("footing_id"),
        "column_id": footing.get("column_id"),
        "size": {
            "step_1": {
                "width": step1.get("width"),
                "depth": step1.get("depth"),
                "length": step1.get("length")
            },
            "step_2": {
                "width": step2.get("width"),
                "depth": step2.get("depth"),
                "length": step2.get("length")
            }
        },
        "reinforcement": {
            "dia": dia,
            "spacing": spacing
        },
        "nos": None,
        "mix": footing.get("mix"),
        "steel_grade": None
    }



# ===========================
# Process PDF
# ===========================
def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = load_prompt()
    all_footings = []

    for img_path in tqdm(image_paths):

        result = extract_from_image(img_path, prompt)

        try:
            parsed = json.loads(result)

            if "footings" in parsed:
                for f in parsed["footings"]:
                    cleaned = clean_footing(f)
                    all_footings.append(cleaned)

        except Exception:
            print("⚠ JSON parse failed")
            print(result)

    output_data = {"footings": all_footings}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Output saved to {output_file}")


# ===========================
# Main
# ===========================
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
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
