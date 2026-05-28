import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image, extract_with_tools


# ===============================
# LOAD PROMPT
# ===============================
def load_prompt():
    prompt_path = os.path.join(
        os.path.dirname(__file__),
        "prompt_2.txt"
    )

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


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

        result = extract_with_tools(img_path, prompt)

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

        # Assign sequential footing IDs
        for footing in footings:

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
