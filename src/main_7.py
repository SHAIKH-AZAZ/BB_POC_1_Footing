import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools
from prompt_builder import build_prompt


def load_prompt():
    return build_prompt(7)


def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
        return None


def clean_footing(footing):

    step1 = footing.get("size", {}).get("step_1", {})
    step2 = footing.get("size", {}).get("step_2", {})
    reinf = footing.get("reinforcement", {})

    # Remove duplicates while preserving order
    dia = list(dict.fromkeys(reinf.get("dia", [])))
    spacing = list(dict.fromkeys(reinf.get("spacing", [])))

    # Normalize spacing format
    spacing = [s.upper().replace("C/C", "C/C") for s in spacing]

    return {
        "footing_id": footing.get("footing_id"),
        "column_id": None,
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
        parsed = safe_json_parse(result)

        if parsed and "footings" in parsed:
            for f in parsed["footings"]:
                cleaned = clean_footing(f)
                all_footings.append(cleaned)
        else:
            print("⚠ JSON parse failed")

    output_data = {"footings": all_footings}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Output saved to {output_file}")


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
