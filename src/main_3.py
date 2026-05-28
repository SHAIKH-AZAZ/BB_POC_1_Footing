import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools


# ===============================
# LOAD PROMPT
# ===============================

def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_3.txt"), "r") as f:
        return f.read()


# ===============================
# EXPAND COLUMN GROUPS
# ===============================

def expand_columns(text):
    """
    Converts:
    AC1,2,11 AC48,49
    →
    AC1,AC2,AC11,AC48,AC49
    """

    if not text:
        return None

    text = text.replace("\n", " ").replace(" ", "")

    matches = re.findall(r'([A-Z]+)([0-9,]+)', text)

    result = []

    for prefix, numbers in matches:
        nums = numbers.split(",")
        for n in nums:
            if n.isdigit():
                result.append(f"{prefix}{n}")

    return ",".join(result) if result else None


# ===============================
# PROCESS PDF
# ===============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = load_prompt()
    final_footings = []
    footing_counter = 1

    for img_path in tqdm(image_paths):

        result = extract_with_tools(img_path, prompt)

        # Handle dict or string safely
        if isinstance(result, dict):
            parsed = result
        else:
            try:
                parsed = json.loads(result)
            except:
                print("⚠ JSON parse failed")
                continue

        for footing in parsed.get("footings", []):

            raw_column = footing.get("column_id", "")

            if isinstance(raw_column, dict):
                raw_column = " ".join(str(v) for v in raw_column.values())
            else:
                raw_column = str(raw_column)

            expanded_columns = expand_columns(raw_column)

            dia = sorted(list(set(footing.get("reinforcement", {}).get("dia", []))))
            spacing = sorted(list(set(footing.get("reinforcement", {}).get("spacing", []))))

            final_footings.append({
                "footing_id": str(footing_counter),
                "column_id": expanded_columns,
                "size": footing.get("size"),
                "reinforcement": {
                    "dia": dia,
                    "spacing": spacing
                },
                "nos": footing.get("nos"),
                "mix": footing.get("mix"),
                "steel_grade": footing.get("steel_grade")
            })

            footing_counter += 1

    output_data = {"footings": final_footings}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
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
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
