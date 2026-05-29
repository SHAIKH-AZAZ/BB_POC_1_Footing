"""
pdf_text_extractor.py - Footing project

Reads exact text from a PDF's text layer using pdfplumber. Coordinates are
expressed in normalized [0, 1] space so the model can reuse the same bbox
format it already uses for ``zoom_region`` calls.

For scanned PDFs (no embedded text), ``has_text_layer`` returns False so
callers can fall back to vision-based reading instead of getting silent
empty strings.
"""

import os
import threading

try:
    import pdfplumber
except ImportError as exc:
    pdfplumber = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_NO_TEXT_LAYER = "<no text layer>"


def _clamp_unit(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    return max(0.0, min(1.0, value))


def _normalize_bbox(x1, y1, x2, y2):
    """Return a sorted, clamped, non-degenerate [0,1] bbox."""

    nx1 = _clamp_unit(x1, 0.0)
    ny1 = _clamp_unit(y1, 0.0)
    nx2 = _clamp_unit(x2, 1.0)
    ny2 = _clamp_unit(y2, 1.0)
    if nx2 < nx1:
        nx1, nx2 = nx2, nx1
    if ny2 < ny1:
        ny1, ny2 = ny2, ny1
    if nx2 - nx1 < 1e-4:
        nx2 = min(1.0, nx1 + 1e-3)
    if ny2 - ny1 < 1e-4:
        ny2 = min(1.0, ny1 + 1e-3)
    return nx1, ny1, nx2, ny2


class PdfTextExtractor:
    """
    Lazy, thread-safe wrapper around a single pdfplumber.PDF.

    pdfplumber's coordinate origin is top-left for crop/extract_text, which
    matches the [0, 1] convention used by ``zoom_region``. We open the PDF
    once per extraction run and reuse pages across calls because opening is
    the expensive part.
    """

    def __init__(self, pdf_path):
        if pdfplumber is None:
            raise RuntimeError(
                f"pdfplumber is required but failed to import: {_IMPORT_ERROR}"
            )
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)
        self.pdf_path = pdf_path
        self._lock = threading.Lock()
        self._pdf = None
        self._text_layer_cache = {}

    def _ensure_open(self):
        if self._pdf is None:
            self._pdf = pdfplumber.open(self.pdf_path)
        return self._pdf

    def close(self):
        with self._lock:
            if self._pdf is not None:
                try:
                    self._pdf.close()
                finally:
                    self._pdf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _get_page(self, page_index):
        pdf = self._ensure_open()
        if page_index < 0 or page_index >= len(pdf.pages):
            raise IndexError(
                f"page_index {page_index} out of range for "
                f"{self.pdf_path} ({len(pdf.pages)} pages)"
            )
        return pdf.pages[page_index]

    def page_count(self):
        with self._lock:
            return len(self._ensure_open().pages)

    def page_size(self, page_index):
        """Return (width, height) of the page in PDF points."""

        with self._lock:
            page = self._get_page(page_index)
            return float(page.width), float(page.height)

    def has_text_layer(self, page_index):
        """
        True iff the requested page exposes any extractable characters.

        Result is cached per page because pdfplumber re-walks the page
        every time ``chars`` is touched.
        """

        with self._lock:
            if page_index in self._text_layer_cache:
                return self._text_layer_cache[page_index]
            page = self._get_page(page_index)
            try:
                has_text = bool(page.chars)
            except Exception:
                has_text = False
            self._text_layer_cache[page_index] = has_text
            return has_text

    def extract_text(self, page_index, x1, y1, x2, y2):
        """
        Read the text inside the normalized [0, 1] bbox on ``page_index``.

        Returns the special sentinel ``"<no text layer>"`` when the page is
        a scan. Returns an empty string when the page has text but the
        cropped region happens to be empty.
        """

        with self._lock:
            if not self.has_text_layer(page_index):
                return _NO_TEXT_LAYER

            page = self._get_page(page_index)
            nx1, ny1, nx2, ny2 = _normalize_bbox(x1, y1, x2, y2)
            w = float(page.width)
            h = float(page.height)
            bbox = (nx1 * w, ny1 * h, nx2 * w, ny2 * h)
            try:
                cropped = page.crop(bbox, relative=False, strict=False)
                text = cropped.extract_text() or ""
            except Exception:
                # Defensive: a malformed page should not crash extraction.
                text = ""
            return text.strip()


def is_no_text_layer(value):
    return value == _NO_TEXT_LAYER
