import os
import json
import re
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools


def load_prompt():
    path = os.path.join(os.path.dirname(__file__), "prompt_9.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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


def remove_duplicates_preserve_order(lst):
    seen = set()
    cleaned = []
    for item in lst:
        if item not in seen:
            cleaned.append(item)
            seen.add(item)
    return cleaned


def clean_footing(f):

    size = f.get("size", {})
    reinf = f.get("reinforcement", {})

    dia = remove_duplicates_preserve_order(reinf.get("dia", []))
    spacing = [s.upper() for s in reinf.get("spacing", [])]
    spacing = remove_duplicates_preserve_order(spacing)

    return {
        "footing_id": f.get("footing_id"),
        "column_id": None,
        "size": {
            "width": size.get("width"),
            "depth": size.get("depth"),
            "length": size.get("length")
        },
        "reinforcement": {
            "dia": dia,
            "spacing": spacing
        },
        "nos": None,
        "mix": None,
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
            for footing in parsed["footings"]:
                cleaned = clean_footing(footing)
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
