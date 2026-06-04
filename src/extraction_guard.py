import json
import os
import re
from datetime import datetime, timezone


ALLOWED_REGION_PURPOSES = {
    "header",
    "row_label",
    "data_cell",
    "global_note",
    "ambiguous_text",
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()]


def _as_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _spacing(value):
    text = str(value).upper().strip()
    if not text:
        return None
    if re.fullmatch(r"\d{2,4}", text):
        return f"{text} C/C"
    text = text.replace("CC", "C/C")
    text = re.sub(r"\s*C\s*/?\s*C\s*", " C/C", text)
    m = re.search(r"(\d{2,4})\s*C/C", text)
    if m:
        return f"{m.group(1)} C/C"
    return None


def extract_rebar_parts(*texts, keep_repetition=False):
    """
    Parse rebar diameter and spacing from free-form schedule text.

    Handles three diameter notations seen across footing patterns:
        - T-first       e.g. "T16", "T 12"           (patterns 1, 3, 5, 7, 10)
        - Number-first  e.g. "16T", "12T-100C/C"     (patterns 8, 9)
        - Bare number near a unit/spacing context    (patterns 2, fallback)

    Spacing is normalized to "<n> C/C" regardless of input separator
    ("@100", "-100C/C", "100 c/c").

    Number-first diameters keep their written orientation ("16T" stays "16T")
    because pattern 8/9 rules explicitly require it. T-first inputs stay
    T-first ("T16"). Both shapes can coexist in the output list.
    """

    dia = []
    spacing = []

    def add_value(target, value):
        if keep_repetition or value not in target:
            target.append(value)

    for text in texts:
        for item in _as_list(text):
            clean = str(item).upper().replace("PHI", "T").replace("Ø", "T")

            # Count-diameter TOR format (pattern 2), e.g. "25-10 TOR" -> "25-10T",
            # "31-12 TOR" -> "31-12T". Must run first and consume the token so the
            # bare-number rules below do not also fire on its digits.
            matched_tor = False
            for dm in re.finditer(r"\b(\d{1,3})\s*-\s*(\d{1,2})\s*TOR\b", clean):
                add_value(dia, f"{dm.group(1)}-{dm.group(2)}T")
                matched_tor = True
            if matched_tor:
                continue

            # Number-first diameter, e.g. "16T" or "12T-100C/C". Must run
            # before the bare-number fallback so "16T" is not misread as the
            # number 16 and dropped.
            for dm in re.finditer(r"\b(\d{1,2})\s*T\b", clean):
                add_value(dia, f"{dm.group(1)}T")

            # T-first diameter, e.g. "T16", "T 12".
            for dm in re.finditer(r"\bT\s*(\d{1,2})\b", clean):
                add_value(dia, f"T{dm.group(1)}")

            # Bare number followed by a spacing/unit cue, e.g. "12 @ 100 C/C".
            # Skip if a T-first or number-first value already covers it.
            for dm in re.finditer(r"\b(\d{1,2})\b(?=\s*(?:@|C/C|MM|$))", clean):
                value = f"T{dm.group(1)}"
                if f"{dm.group(1)}T" in dia or (not keep_repetition and value in dia):
                    continue
                add_value(dia, value)

            # Spacing values, e.g. "@150", "-150C/C", "150 c/c".
            for sm in re.finditer(r"(?:@|-)?\s*(\d{2,4})\s*C\s*/?\s*C", clean):
                add_value(spacing, f"{sm.group(1)} C/C")
            for sm in re.finditer(r"@\s*(\d{2,4})", clean):
                add_value(spacing, f"{sm.group(1)} C/C")

    return {"dia": dia, "spacing": spacing}


def expand_column_group(text):
    """
    Expand grouped column marks like "AC1,2,11 AC48,49" into the explicit
    list "AC1,AC2,AC11,AC48,AC49". Returns None when no prefix-number
    pattern is found.
    """

    if text is None:
        return None
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values())
    text = str(text).replace("\n", " ").replace(" ", "")
    if not text:
        return None

    result = []
    for prefix, numbers in re.findall(r"([A-Z]+)([0-9,\-]+)", text):
        for num in numbers.split(","):
            num = num.strip()
            if not num:
                continue
            if num.isdigit():
                result.append(f"{prefix}{num}")
            elif re.fullmatch(r"\d+-\d+", num):
                # Preserve compound IDs like "230-1".
                result.append(f"{prefix}{num}")
    return ",".join(result) if result else None


def build_column_record(args):
    return {
        "column_no": args.get("column_no", ""),
        "column_name": args.get("storey_level", ""),
        "size": {
            "width": args.get("width"),
            "depth": args.get("depth"),
            "length": None,
        },
        "reinforcement": args.get("reinforcement", []) or [],
        "stirrups": {
            "dia": [args["ties_dia"]] if args.get("ties_dia") else [],
            "spacing": [args["ties_spacing"]] if args.get("ties_spacing") else [],
        },
        "mix": args.get("mix"),
        "steel_grade": args.get("steel_grade"),
    }


def build_slab_record(args):
    along = args.get("steel_along_span", []) or []
    across = args.get("steel_across_span", []) or []
    reinf = extract_rebar_parts(along, across)
    record = {
        "slab_id": args.get("slab_id") or args.get("slab_type", ""),
        "thickness": args.get("thickness"),
        "type": args.get("type", "") or "",
        "mix": args.get("mix", "") or "",
        "reinforcement": reinf,
    }
    if args.get("remarks") is not None:
        record["remarks"] = args.get("remarks")
    return record


def build_footing_record(args):
    reinf_texts = [
        args.get("short_span_reinf"),
        args.get("long_span_reinf"),
        args.get("bottom_short_reinf"),
        args.get("bottom_long_reinf"),
        args.get("top_short_reinf"),
        args.get("top_long_reinf"),
        args.get("bottom_steel"),
        args.get("top_steel"),
        args.get("main_bar"),
        args.get("distribution_bar"),
        args.get("reinforcement"),
    ]
    stirrup_texts = [
        args.get("stirrup_reinf"),
        args.get("stirrups"),
        args.get("ties_dia"),
        args.get("ties_spacing"),
        args.get("links"),
        args.get("link_spacing"),
    ]
    reinf = extract_rebar_parts(*reinf_texts, keep_repetition=True)
    stirrups = extract_rebar_parts(*stirrup_texts, keep_repetition=True)
    column_id = args.get("column_id", "")
    record = {
        "footing_id": args.get("footing_id") or column_id,
        "column_id": column_id,
        "size": {
            "width": args.get("plan_width"),
            "depth": args.get("depth_bottom")
            if args.get("depth_bottom") is not None
            else args.get("depth_top"),
            "length": args.get("plan_length"),
        },
        "reinforcement": reinf,
        "nos": args.get("nos"),
        "mix": _dedupe(_as_list(args.get("mix") or args.get("concrete_mix"))),
        "steel_grade": args.get("steel_grade"),
    }
    if stirrups["dia"] or stirrups["spacing"]:
        record["stirrups"] = stirrups
    # Optional raw cell text for downstream verification (pattern 2 refer-plan check).
    if args.get("footing_cell_text"):
        record["footing_cell_text"] = args.get("footing_cell_text")
    return record


class ExtractionState:
    def __init__(
        self,
        project,
        image_path,
        output_key,
        duplicate_key_fields,
        pdf_extractor=None,
        page_index=0,
        enforce_zoom=False,
    ):
        self.project = project
        self.image_path = image_path
        self.output_key = output_key
        self.duplicate_key_fields = duplicate_key_fields
        self.pdf_extractor = pdf_extractor
        self.page_index = page_index
        self.enforce_zoom = enforce_zoom
        self.think_seen = False
        self.think = None
        self.zoom_regions = {}
        self.confirmed_reads = {}
        self.records = []
        self.record_sources = []
        self.warnings = []
        self.pdf_reads = []  # audit trail for read_pdf_text calls

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)
        print(f"  [WARN] {message}")

    def handle_think(self, args):
        self.think_seen = True
        self.think = dict(args)
        missing = [
            key
            for key in (
                "image_quality",
                "table_bounds",
                "expected_count",
                "zoom_plan",
                "normalization_rules",
                "extraction_order",
            )
            if key not in args
        ]
        if missing:
            self.warn(f"think missing structured field(s): {', '.join(missing)}")
        expected = self.expected_count()
        return (
            "Structured extraction plan accepted. "
            f"Expected record count: {expected if expected is not None else 'unknown'}. "
            "Now call zoom_region for planned regions, then confirm_read for each important read, "
            f"then call add_{self.project} with source_region_ids."
        )

    def handle_zoom(self, args):
        if not self.think_seen:
            self.warn("zoom_region rejected before think")
            return None, "Call think first; zoom_region is only available after the structured plan."

        region_id = str(args.get("region_id") or f"region_{len(self.zoom_regions) + 1}").strip()
        purpose = str(args.get("purpose") or "ambiguous_text").strip()
        if purpose not in ALLOWED_REGION_PURPOSES:
            self.warn(f"zoom_region purpose '{purpose}' is not recognized")
            purpose = "ambiguous_text"

        try:
            x1 = float(args.get("x1", 0.0))
            y1 = float(args.get("y1", 0.0))
            x2 = float(args.get("x2", 1.0))
            y2 = float(args.get("y2", 1.0))
        except (TypeError, ValueError):
            return None, "x1/y1/x2/y2 must be numbers between 0.0 and 1.0."

        # Coordinates must be normalized fractions. If the model sends pixel-like
        # values (e.g. 800), reject with guidance instead of silently clamping to
        # a degenerate strip that reads nothing.
        if any(v < 0.0 or v > 1.0 for v in (x1, y1, x2, y2)):
            self.warn(f"zoom_region coords out of range: ({x1},{y1})-({x2},{y2})")
            return None, (
                "Coordinates must be NORMALIZED FRACTIONS between 0.0 and 1.0, not pixels. "
                "(0,0) = top-left, (1,1) = bottom-right. "
                "Example for the bottom footing band: x1=0.0, y1=0.82, x2=1.0, y2=0.98. "
                "Re-call zoom_region with fractional coordinates."
            )

        region = {
            "region_id": region_id,
            "purpose": purpose,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "reason": args.get("reason", ""),
        }
        self.zoom_regions[region_id] = region
        return region, (
            f"Zoomed region '{region_id}' accepted. "
            "Read it carefully and call confirm_read with the exact text before adding records."
        )

    def handle_confirm_read(self, args):
        if not self.think_seen:
            self.warn("confirm_read rejected before think")
            return False, "Call think first, then zoom_region, then confirm_read."

        region_id = str(args.get("region_id") or "").strip()
        if region_id not in self.zoom_regions:
            self.warn(f"confirm_read rejected for unzoomed region '{region_id}'")
            return False, f"Region '{region_id}' has not been zoomed. Call zoom_region first."

        text = str(args.get("text") or "").strip()
        if not text:
            self.warn(f"confirm_read for region '{region_id}' is empty")

        read = {
            "region_id": region_id,
            "text": text,
            "confidence": args.get("confidence"),
            "applies_to": args.get("applies_to"),
        }
        self.confirmed_reads[region_id] = read
        return True, f"Confirmed read for region '{region_id}' recorded."

    def _image_longest_side(self):
        if getattr(self, "_img_longest", None) is None:
            self._img_longest = 0
            try:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                with Image.open(self.image_path) as im:
                    self._img_longest = max(im.size)
            except Exception:
                self._img_longest = 0
        return self._img_longest

    def can_add_record(self, args):
        if not self.think_seen:
            self.warn(f"add_{self.project} rejected before think")
            return False, f"Call think before add_{self.project}."

        # On very large sheets (e.g. 94x66 in CAD exports), the full image is
        # downscaled by the vision API until the footing band is unreadable, so
        # the model copies values from the wrong rows. When enforce_zoom is set,
        # require at least one confirm_read (which forces a high-res zoom crop)
        # before accepting any record. Opt-in per pattern to avoid disrupting
        # patterns whose table already fills the sheet.
        if self.enforce_zoom and self.project == "footing" and self._image_longest_side() > 6000:
            if not self.confirmed_reads:
                self.warn("add_footing rejected on large image before any zoom/confirm_read")
                return False, (
                    "This drawing is very large and the full image is too downscaled to read "
                    "the footing schedule accurately. Before adding any footing you MUST:\n"
                    "1. zoom_region into the FOOTING schedule band (the rows FOOTING SIZE, "
                    "DEPTH (d TO D), REINF. //B, REINF. //L, FOOTING MIX., and the COL. MARK row).\n"
                    "2. zoom_region into the specific column you are about to record.\n"
                    "3. confirm_read the exact text you see in that zoomed crop.\n"
                    "Then call add_footing using ONLY the values you confirmed. "
                    "Do NOT reuse the same reinforcement for every column."
                )
        return True, "Record accepted."

    def handle_read_pdf_text(self, args):
        """
        Read text from the source PDF's text layer for an arbitrary [0,1]
        bbox. Returns ``(text, message)`` where ``text`` is None when the
        read is unavailable (no PDF bound, no text layer, or before think).
        """

        if not self.think_seen:
            self.warn("read_pdf_text rejected before think")
            return None, "Call think before read_pdf_text."

        if self.pdf_extractor is None:
            return None, (
                "PDF text reading is not available for this run. "
                "Use zoom_region + confirm_read instead."
            )

        try:
            x1 = float(args.get("x1", 0.0))
            y1 = float(args.get("y1", 0.0))
            x2 = float(args.get("x2", 1.0))
            y2 = float(args.get("y2", 1.0))
        except (TypeError, ValueError):
            return None, "x1/y1/x2/y2 must be numeric."

        try:
            text = self.pdf_extractor.extract_text(self.page_index, x1, y1, x2, y2)
        except Exception as exc:
            self.warn(f"read_pdf_text failed: {exc}")
            return None, f"PDF text read failed: {exc}"

        entry = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "purpose": str(args.get("purpose") or "ambiguous_text"),
            "reason": args.get("reason", ""),
            "text": text,
        }
        self.pdf_reads.append(entry)
        return text, "OK"

    def add_record(self, record, args):
        self.records.append(record)
        self.record_sources.append({
            "record": record,
            "source_region_ids": _as_list(args.get("source_region_ids")),
        })

    def expected_count(self):
        if not self.think:
            return None
        explicit = _as_int(self.think.get("expected_count"))
        if explicit is not None:
            return explicit
        if self.project == "column":
            headers = self.think.get("column_headers") or []
            levels = self.think.get("storey_levels") or []
            if headers and levels:
                return len(headers) * len(levels)
        if self.project == "slab":
            slab_ids = self.think.get("slab_ids") or []
            if slab_ids:
                return len(slab_ids)
        if self.project == "footing":
            groups = self.think.get("column_groups") or []
            if groups:
                return len(groups)
        return None

    def validate(self, records=None):
        records = records if records is not None else self.records
        expected = self.expected_count()
        if expected is not None and expected != len(records):
            self.warn(f"expected {expected} record(s), collected {len(records)}")

        seen = {}
        for record in records:
            key = tuple(str(record.get(field, "")) for field in self.duplicate_key_fields)
            if key in seen:
                self.warn(f"duplicate record key detected: {key}")
            seen[key] = True

        if self.project == "column":
            for record in records:
                column_no = str(record.get("column_no", "")).replace(" ", "")
                if "C7,C75" in column_no and "C67" not in column_no:
                    self.warn(
                        f"suspicious column header '{record.get('column_no')}' may have lost C67"
                    )

        trace_text = json.dumps(self.think or {}) + " " + json.dumps(
            list(self.confirmed_reads.values())
        )
        record_text = json.dumps(records)
        for match in re.findall(r"\b(?:1[0-9]|2[0-9]|3[0-9])-T\d+\b", trace_text):
            if match not in record_text:
                self.warn(f"confirmed two-digit reinforcement '{match}' not found in records")

    def trace_payload(self):
        return {
            "project": self.project,
            "image_path": self.image_path,
            "page_index": self.page_index,
            "pdf_path": getattr(self.pdf_extractor, "pdf_path", None),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "think": self.think,
            "zoom_regions": list(self.zoom_regions.values()),
            "confirmed_reads": list(self.confirmed_reads.values()),
            "pdf_reads": list(self.pdf_reads),
            "records": self.records,
            "added_records": self.record_sources,
            "expected_count": self.expected_count(),
            "actual_count": len(self.records),
            "warnings": self.warnings,
        }

    def write_trace(self):
        folder = os.path.dirname(os.path.abspath(self.image_path))
        stem = os.path.basename(folder) or os.path.splitext(os.path.basename(self.image_path))[0]
        path = os.path.join(folder, f"{stem}_trace.json")
        page_key = os.path.basename(self.image_path)

        existing = {
            "project": self.project,
            "pdf_stem": stem,
            "pages": {},
            "summary": {},
        }
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing.update(loaded)
                    existing.setdefault("pages", {})
            except (OSError, json.JSONDecodeError):
                self.warn(f"could not read existing trace file: {path}")

        existing["pages"][page_key] = self.trace_payload()
        all_warnings = []
        total_records = 0
        for page in existing["pages"].values():
            total_records += int(page.get("actual_count") or 0)
            all_warnings.extend(page.get("warnings") or [])
        existing["summary"] = {
            "page_count": len(existing["pages"]),
            "record_count": total_records,
            "warning_count": len(all_warnings),
            "warnings": _dedupe(all_warnings),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return path


def clean_json_string(text):
    text = str(text or "").strip()
    if "```" in text:
        parts = text.split("```")
        block = parts[1] if len(parts) >= 3 else parts[0]
        if block.lower().startswith("json"):
            block = block[4:]
        text = block.strip()
    return text
