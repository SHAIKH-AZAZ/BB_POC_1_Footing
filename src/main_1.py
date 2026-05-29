import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools
from prompt_builder import build_prompt


def load_prompt():
    return build_prompt(1)


def normalize_column_id(column_id):
    if not column_id:
        return column_id

    column_id = column_id.replace("CC", "GC")
    column_id = column_id.replace("FCC", "")
    column_id = column_id.replace("ECC", "")
    return column_id.strip()


def normalize_steel_grade(steel):
    if not steel:
        return steel

    steel = steel.upper()
    steel = steel.replace("FES00", "FE500")
    steel = steel.replace("FE50O", "FE500")
    return steel


def clean_footing(footing):

    # Deduplicate reinforcement
    dia = footing.get("reinforcement", {}).get("dia", [])
    spacing = footing.get("reinforcement", {}).get("spacing", [])

    footing["reinforcement"]["dia"] = sorted(list(set(dia)))
    footing["reinforcement"]["spacing"] = sorted(list(set(spacing)))

    footing["column_id"] = normalize_column_id(
        footing.get("column_id", "")
    )

    footing["steel_grade"] = normalize_steel_grade(
        footing.get("steel_grade", None)
    )

    return footing


def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

    prompt = load_prompt()
    all_footings = []

    for img_path in tqdm(image_paths):
        result = extract_with_tools(img_path, prompt)

        try:
            parsed = json.loads(result)
            if "footings" in parsed:
                all_footings.extend(parsed["footings"])
        except Exception:
            print("⚠ JSON parse failed")
            print(result)

    cleaned = [clean_footing(f) for f in all_footings]

    final_output = {"footings": cleaned}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Output saved to {output_file}")


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
