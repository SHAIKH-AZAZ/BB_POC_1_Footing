"""
pdf_to_images.py - Footing project

Renders PDF pages to PNG.

CAD / AutoCAD-exported drawings frequently use very thin strokes in light
pastel colors (pale green, pink, cyan) on a white background. A plain raster
render keeps those colors at their true high luminance, so the text comes out
almost invisible and the vision model cannot read it reliably.

To fix that we run an "ink boosting" pass on each rendered page:

  ink = 255 - min(R, G, B)

White background pixels (R=G=B=255) give ink = 0. ANY colored stroke drops at
least one channel below 255, so it produces ink > 0 regardless of hue. We then
amplify that ink and paint it as black-on-white, which makes faint pastel CAD
text dark and readable without depending on luminance (a luminance grayscale
would treat pale pink/green as near-white and lose it).
"""

import os

import fitz
import numpy as np
from PIL import Image


def _pixmap_to_array(pix):
    """Convert a fitz Pixmap to an (H, W, 3) uint8 RGB numpy array."""
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:          # RGBA -> drop alpha
        arr = arr[:, :, :3]
    elif pix.n == 1:        # grayscale -> RGB
        arr = np.repeat(arr, 3, axis=2)
    return arr


def enhance_ink(rgb, ink_floor=6, ink_knee=30):
    """
    Boost faint colored CAD strokes into dark, readable ink on white.

    CAD pastel strokes are extremely faint: on a typical export the darkest
    stroke pixel only reaches an "ink" value (255 - min channel) of ~30, and
    header text sits around ~25. A flat multiplier can't both clean the
    background and darken text, so we apply a levels / contrast stretch:

        ink <= ink_floor   -> white (treated as background / AA haze)
        ink >= ink_knee    -> black (full strength ink)
        in between         -> linearly mapped

    With the defaults, faint header strokes (ink ~25-30) map to near-black
    while the 98%+ white background stays pure white.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    ink_floor : int
        Ink level at/below which a pixel is background. Raise it if faint
        compression noise leaks through; lower it if very light strokes are
        being dropped.
    ink_knee : int
        Ink level mapped to full black. Lower = more aggressive darkening.

    Returns
    -------
    np.ndarray
        (H, W) uint8 grayscale: black ink on white background.
    """
    rgb = rgb.astype(np.int16)
    min_channel = rgb.min(axis=2)               # near 255 on white, lower on any stroke
    ink = 255 - min_channel                     # 0 on white, positive on strokes

    span = max(1, ink_knee - ink_floor)
    strength = np.clip((ink - ink_floor) / span, 0.0, 1.0)
    gray = (1.0 - strength) * 255.0             # strength 1 -> black, 0 -> white
    return gray.astype(np.uint8)


def convert_pdf_to_images(pdf_path, output_folder, dpi=600, enhance=True,
                          ink_floor=6, ink_knee=30):
    """
    Render each page of ``pdf_path`` to a PNG in ``output_folder``.

    When ``enhance`` is True (default), faint pastel CAD line work is boosted
    into dark, high-contrast text via a levels stretch (see ``enhance_ink``).
    Set ``enhance=False`` to keep the original raw render.
    """
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    try:
        for page_number in range(len(doc)):
            page = doc[page_number]
            pix = page.get_pixmap(dpi=dpi)

            image_path = os.path.join(
                output_folder,
                f"page_{page_number + 1}.png",
            )

            if enhance:
                rgb = _pixmap_to_array(pix)
                gray = enhance_ink(rgb, ink_floor=ink_floor, ink_knee=ink_knee)
                Image.fromarray(gray, mode="L").save(image_path)
            else:
                pix.save(image_path)

            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths
