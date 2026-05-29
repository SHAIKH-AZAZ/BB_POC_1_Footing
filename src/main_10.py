import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools
from prompt_builder import build_prompt


def load_prompt():
    return build_prompt(10)


def clean_reinforcement(reinf):

    dia = []
    spacing = []

    for d in reinf.get("dia", []):
        if d and d not in dia:
            dia.append(d)

    for s in reinf.get("spacing", []):
        if s and s not in spacing:
            spacing.append(s)

    return {
        "dia": dia,
        "spacing": spacing
    }


def clean_footing(footing):

    return {
        "footing_id": footing.get("footing_id"),
        "column_id": footing.get("column_id"),
        "size": {
            "width": footing.get("size", {}).get("width"),
            "depth": footing.get("size", {}).get("depth"),
            "length": footing.get("size", {}).get("length")
        },
        "reinforcement": clean_reinforcement(
            footing.get("reinforcement", {})
        ),
        "nos": footing.get("nos"),
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

        try:
            parsed = json.loads(result)
            if "footings" in parsed:
                for f in parsed["footings"]:
                    all_footings.append(clean_footing(f))
        except:
            print("⚠ JSON parse failed")
            print(result)

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
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()