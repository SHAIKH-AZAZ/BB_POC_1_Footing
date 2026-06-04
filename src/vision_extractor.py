"""
vision_extractor.py - Footing project

Tool-augmented extraction with enforced sequence:
think -> zoom_region -> confirm_read -> add_footing.
"""

import base64
import io
import json
import os
import time

from PIL import Image
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_IMAGE_DETAIL, OPENAI_MODEL
from extraction_guard import ExtractionState, build_footing_record, clean_json_string


client = OpenAI(api_key=OPENAI_API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def _image_content(base64_image, detail=OPENAI_IMAGE_DETAIL):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}",
            "detail": detail,
        },
    }


def _crop_image_b64(image_path, x1, y1, x2, y2):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))
    if x2 <= x1:
        x2 = min(1.0, x1 + 0.05)
    if y2 <= y1:
        y2 = min(1.0, y1 + 0.05)
    left = int(x1 * w)
    top = int(y1 * h)
    right = max(int(x2 * w), left + 20)
    bottom = max(int(y2 * h), top + 20)
    cropped = img.crop((left, top, right, bottom))
    longest = max(cropped.size)
    if longest < 1200:
        scale = 1200 / longest
        cropped = cropped.resize(
            (int(cropped.width * scale), int(cropped.height * scale)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_REGION_PURPOSE_ENUM = [
    "header",
    "row_label",
    "data_cell",
    "global_note",
    "ambiguous_text",
]


def _with_tool_protocol(prompt_text):
    return (
        "MIX & STEEL GRADE EXTRACTION RULE — READ THIS CAREFULLY:\n"
        "Concrete grades and steel grades may appear as vertical labels, side labels, merged cells, row headers, or notes within the FOOTING DETAILS table.\n\n"

        "1) COMBINED CONCRETE + STEEL LABELS\n"
        "   Examples:\n"
        "     M25 : Fe500\n"
        "     M30 : Fe500D\n"
        "     M35 : Fe550\n"
        "   Interpretation:\n"
        "     - M25 / M30 / M35 = CONCRETE MIX GRADE\n"
        "     - Fe500 / Fe500D / Fe550 = STEEL GRADE\n"
        "   Extraction Rules:\n"
        "     - ALWAYS extract BOTH values.\n"
        "     - ALWAYS add Mxx values to the mix array.\n"
        "     - ALWAYS add Fexxx values to the steel_grade array.\n"
        "     - NEVER ignore the Mxx value because it appears together with a steel grade.\n\n"

        "2) PCC CONCRETE LABELS\n"
        "   Examples:\n"
        "     M10\n"
        "     M15\n"
        "     M20\n"
        "   when associated with P.C.C. / PCC rows.\n"
        "   Extraction Rules:\n"
        "     - These are ALSO concrete mix grades.\n"
        "     - Add them to the same mix array.\n"
        "     - Do not overwrite previously found mix grades.\n\n"

        "3) MULTIPLE MIX GRADES ARE VALID\n"
        "   A footing may contain:\n"
        "     - Structural Concrete = M25\n"
        "     - PCC Concrete = M15\n"
        "   Both MUST be captured.\n\n"

        "4) REQUIRED VALIDATION\n"
        "   If 'Mxx : Fexxx' is detected anywhere in the footing table:\n"
        "     - Mxx MUST appear in mix.\n"
        "     - Fexxx MUST appear in steel_grade.\n"
        "   Missing the Mxx value is an extraction error.\n\n"

        "5) DEDUPLICATION\n"
        "   Store only unique values.\n"
        "   Example:\n"
        "     mix = ['M25', 'M15']\n"
        "     steel_grade = ['Fe500']\n\n"

        "6) SEARCH AREA\n"
        "   Look for mix and steel grades in:\n"
        "     - Left side labels\n"
        "     - Right side labels\n"
        "     - Vertical text\n"
        "     - Row headers\n"
        "     - Table notes\n"
        "     - Merged cells\n"
        "     - Top and bottom annotations\n"
        "   Do NOT limit the search to a single side of the table.\n\n"

        f"{prompt_text}"
    )


FOOTING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Mandatory first call. First locate the FOOTING DETAILS table on the drawing sheet "
                "(ignore all floor column sections, cross-sections, and other tables). "
                "Then return structured planning data: column mark groups from the COLUMN MARK row "
                "(e.g. C1,C18 / C2,C9 — NOT floor labels like '4TH FLOOR COLUMN'), "
                "row structure, global notes, expected count, and zoom targets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_quality": {"type": "string"},
                    "table_bounds": {"type": "object"},
                    "column_groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All column mark groups left to right.",
                    },
                    "row_structure": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Confirmed schedule rows such as COLUMN MARK, SIZE, D TO d, SHORT SPAN, LONG SPAN. ETC can be given ",
                    },
                    "global_notes": {
                        "type": "object",
                        "description": "Concrete mix and steel grade notes found anywhere in the table.",
                    },
                    "concrete_grades": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ALL concrete mix grades (M-grades) found anywhere in the table — "
                            "in vertical labels, row headers, merged cells, or notes. "
                            "IMPORTANT: A label like 'M25 : Fe500' contains M25 as a concrete grade. "
                            "A label like 'M15' next to PCC row is also a concrete grade. "
                            "List ALL: e.g. ['M25', 'M15']. This list will be used for every add_footing call."
                        ),
                    },
                    "steel_grades": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ALL steel grades (Fe-grades) found anywhere in the table. "
                            "A label like 'M25 : Fe500' contains Fe500 as the steel grade. "
                            "List ALL: e.g. ['Fe500']."
                        ),
                    },
                    "expected_count": {"type": "integer"},
                    "zoom_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "region_id": {"type": "string"},
                                "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                                "target": {"type": "string"},
                                "x1": {"type": "number", "description": "Left edge as a fraction 0.0-1.0 of image width."},
                                "y1": {"type": "number", "description": "Top edge as a fraction 0.0-1.0 of image height."},
                                "x2": {"type": "number", "description": "Right edge as a fraction 0.0-1.0 of image width (must be > x1)."},
                                "y2": {"type": "number", "description": "Bottom edge as a fraction 0.0-1.0 of image height (must be > y1)."},
                                "reason": {"type": "string"},
                            },
                            "required": ["region_id", "purpose", "target", "reason"],
                        },
                    },
                    "normalization_rules": {"type": "array", "items": {"type": "string"}},
                    "extraction_order": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "image_quality",
                    "table_bounds",
                    "column_groups",
                    "row_structure",
                    "global_notes",
                    "expected_count",
                    "zoom_plan",
                    "normalization_rules",
                    "extraction_order",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_region",
            "description": (
                "Crop and zoom a planned footing schedule region after think. "
                "Coordinates are NORMALIZED FRACTIONS of the image in the range 0.0 to 1.0, "
                "NOT pixels. (0,0) is the top-left corner, (1,1) is the bottom-right corner. "
                "Example: the bottom footing band spanning the full width is about "
                "x1=0.0, y1=0.82, x2=1.0, y2=0.98."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                    "x1": {"type": "number", "description": "Left edge as a fraction 0.0-1.0 of image width."},
                    "y1": {"type": "number", "description": "Top edge as a fraction 0.0-1.0 of image height."},
                    "x2": {"type": "number", "description": "Right edge as a fraction 0.0-1.0 of image width (must be > x1)."},
                    "y2": {"type": "number", "description": "Bottom edge as a fraction 0.0-1.0 of image height (must be > y1)."},
                    "reason": {"type": "string"},
                },
                "required": ["region_id", "purpose", "x1", "y1", "x2", "y2", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_read",
            "description": "Record exact text read from a zoomed region before add_footing references it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "string"},
                    "applies_to": {"type": "string"},
                },
                "required": ["region_id", "text", "confidence", "applies_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_footing",
            "description": (
                "Record ONE footing group. Call once per column group (left to right). "
                "Do not batch multiple groups into one call. "
                "zoom_region/confirm_read are optional — use only for blurry or ambiguous cells."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "footing_id": {"type": ["string", "null"]},
                    "column_id": {"type": "string"},
                    "footing_cell_text": {
                        "type": ["string", "null"],
                        "description": (
                            "The EXACT raw text you read in THIS column's footing block "
                            "(the FOOTING SIZE / P.C.C. SIZE / DEPTH / REINF / FOOTING MIX rows for this column only). "
                            "Copy it verbatim, including any 'REFER RAFT FOOTING PLAN' note. "
                            "This is used to verify the column actually has footing data."
                        ),
                    },
                    "plan_length": {
                        "type": ["number", "null"],
                        "description": (
                            "Footing length (second dimension). "
                            "Row may be labeled: 'SIZE', 'B x L', 'B X L', 'FOOTING SIZE', 'L', or similar. "
                            "Example: '1200 x 1500' → 1500. Numbers only, no units."
                        ),
                    },
                    "plan_width": {
                        "type": ["number", "null"],
                        "description": (
                            "Footing width (first dimension). "
                            "Row may be labeled: 'SIZE', 'B x L', 'B X L', 'FOOTING SIZE', 'B', or similar. "
                            "Example: '1200 x 1500' → 1200. Numbers only, no units."
                        ),
                    },
                    "depth_top": {
                        "type": ["number", "null"],
                        "description": (
                            "Footing depth. Row may be labeled: 'DEPTH', 'DEPTH (D)', 'D', 'd', 'D TO d', or similar. "
                            "Numbers only, no units."
                        ),
                    },
                    "depth_bottom": {
                        "type": ["number", "null"],
                        "description": (
                            "Footing depth (preferred over depth_top). "
                            "Row may be labeled: 'DEPTH', 'DEPTH (D)', 'D', 'd', 'D TO d', or similar. "
                            "Numbers only, no units."
                        ),
                    },
                    "short_span_reinf": {
                        "type": ["string", "null"],
                        "description": (
                            "Short-side/B-direction reinforcement text from all matching rows under this footing, "
                            "for example '16T-100C/C'. Include bottom and top values comma-separated if both exist."
                        ),
                    },
                    "long_span_reinf": {
                        "type": ["string", "null"],
                        "description": (
                            "Long-side/L-direction reinforcement text from all matching rows under this footing, "
                            "for example '16T-100C/C'. Include bottom and top values comma-separated if both exist."
                        ),
                    },
                    "bottom_short_reinf": {
                        "type": ["string", "null"],
                        "description": "STEEL // - B (SHORT SIDE) (BOTTOM FACE) cell text, if visible.",
                    },
                    "bottom_long_reinf": {
                        "type": ["string", "null"],
                        "description": "STEEL // - L (LONG SIDE) (BOTTOM FACE) cell text, if visible.",
                    },
                    "top_short_reinf": {
                        "type": ["string", "null"],
                        "description": "STEEL // - B (SHORT SIDE) (TOP FACE) cell text, if visible.",
                    },
                    "top_long_reinf": {
                        "type": ["string", "null"],
                        "description": "STEEL // - L (LONG SIDE) (TOP FACE) cell text, if visible.",
                    },
                    "reinforcement": {
                        "type": ["array", "string", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "Any visible reinforcement/bar-spacing text for this footing column. "
                            "Use this as a fallback if the row direction is unclear."
                        ),
                    },
                    "stirrup_reinf": {
                        "type": ["string", "null"],
                        "description": (
                            "Visible stirrup/tie/link/shear reinforcement text for this footing column, "
                            "including diameter and spacing when present."
                        ),
                    },
                    "stirrups": {
                        "type": ["array", "string", "null"],
                        "items": {"type": "string"},
                        "description": "Any visible stirrup, tie, link, or shear reinforcement texts.",
                    },
                    "ties_dia": {
                        "type": ["string", "null"],
                        "description": "Tie/stirrup/link diameter, for example 'T8' or '8T'.",
                    },
                    "ties_spacing": {
                        "type": ["string", "null"],
                        "description": "Tie/stirrup/link spacing, for example '150 C/C'.",
                    },
                    "nos": {"type": ["number", "null"]},
                    "mix": {
                        "type": ["array", "string", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "Use the concrete_grades list you identified in the think phase. "
                            "Copy those exact values here for every add_footing call. "
                            "Example: if think.concrete_grades = ['M25', 'M15'], set mix = ['M25', 'M15']. "
                            "Never leave this empty if concrete_grades was non-empty in think."
                        ),
                    },
                    "steel_grade": {
                        "type": ["array", "string", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "ALL steel grades identified in the think phase (steel_grades array). "
                            "Use the same values you listed in think.steel_grades. "
                            "Example: ['Fe500']."
                        ),
                    },
                    "source_region_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["column_id"],
            },
        },
    },
]


def extract_with_tools(image_path, prompt_text, max_iterations=300, enforce_zoom=False):
    base64_image = encode_image(image_path)
    state = ExtractionState(
        project="footing",
        image_path=image_path,
        output_key="footings",
        duplicate_key_fields=["footing_id", "column_id"],
        enforce_zoom=enforce_zoom,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _with_tool_protocol(prompt_text)},
                _image_content(base64_image),
            ],
        }
    ]
    collected_footings = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    api_call_count = 0

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=FOOTING_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        api_call_count += 1
        usage = response.usage
        if usage:
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            print(
                f"  [tokens] call #{api_call_count}: "
                f"prompt={usage.prompt_tokens}  completion={usage.completion_tokens}  "
                f"total={usage.total_tokens}"
            )

        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            break

        tool_results = []
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "think":
                state.handle_think(args)
                groups = args.get("column_groups") or []
                expected = state.expected_count()
                concrete_grades = args.get("concrete_grades") or []
                steel_grades = args.get("steel_grades") or []
                print("\n============================================================")
                print("  [THINK] Structured footing extraction plan accepted")
                print(f"  groups={len(groups)} expected={expected}")
                print(f"  concrete_grades={concrete_grades}")
                print(f"  steel_grades={steel_grades}")
                print("============================================================\n")
                result_content = (
                    f"Plan accepted. {len(groups)} footing groups = {expected} expected add_footing calls.\n"
                    f"Concrete grades identified: {concrete_grades}. "
                    f"Steel grades identified: {steel_grades}.\n"
                    "For EVERY add_footing call: set mix={concrete_grades} and steel_grade={steel_grades} exactly.\n"
                    "NOW start calling add_footing immediately — one call per footing group.\n"
                    f"Column groups in order: {', '.join(groups)}.\n"
                    "Work left-to-right through ALL groups. Do NOT stop until every group has been recorded. "
                    "Use zoom_region only if a cell is genuinely unclear."
                )
            elif fn_name == "zoom_region":
                region, message = state.handle_zoom(args)
                if not region:
                    result_content = message
                else:
                    print(
                        f"  zoom_region {region['region_id']} "
                        f"({region['x1']:.2f},{region['y1']:.2f})->"
                        f"({region['x2']:.2f},{region['y2']:.2f})"
                    )
                    cropped_b64 = _crop_image_b64(
                        image_path,
                        region["x1"],
                        region["y1"],
                        region["x2"],
                        region["y2"],
                    )
                    result_content = [
                        {
                            "type": "text",
                            "text": (
                                f"{message} Confirm column marks, ECC/FCC suffixes, dimensions, "
                                "reinforcement dia/spacing, and global mix/steel notes exactly."
                            ),
                        },
                        _image_content(cropped_b64),
                    ]
            elif fn_name == "confirm_read":
                _, result_content = state.handle_confirm_read(args)
                print(f"  confirm_read {args.get('region_id', '')}: {str(args.get('text', ''))[:80]}")
            elif fn_name == "add_footing":
                ok, message = state.can_add_record(args)
                if not ok:
                    result_content = message
                else:
                    footing = build_footing_record(args)
                    collected_footings.append(footing)
                    state.add_record(footing, args)
                    expected = state.expected_count() or "?"
                    remaining = (state.expected_count() or 0) - len(collected_footings)
                    result_content = (
                        f"Recorded {len(collected_footings)}/{expected}: footing '{footing['column_id']}'. "
                        f"{remaining} more to go — keep calling add_footing for remaining groups."
                    )
                    size = footing.get("size") or {}
                    reinf = footing.get("reinforcement") or {}
                    print(
                        f"  [extracted] column={footing['column_id']} | "
                        f"size={size.get('width')}x{size.get('length')}x{size.get('depth')} | "
                        f"dia={reinf.get('dia')} spacing={reinf.get('spacing')} | "
                        f"mix={footing.get('mix')} | steel={footing.get('steel_grade')}"
                    )
            else:
                state.warn(f"unknown tool ignored: {fn_name}")
                result_content = f"Unknown tool '{fn_name}' ignored."

            tool_results.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_content}
            )
        messages.extend(tool_results)

    state.validate(collected_footings)
    trace_path = state.write_trace()
    print(f"  Trace saved to {trace_path}")

    total_tokens = total_prompt_tokens + total_completion_tokens
    print(
        f"\n  ── Token usage ({os.path.basename(image_path)}) ──\n"
        f"  API calls      : {api_call_count}\n"
        f"  Prompt tokens  : {total_prompt_tokens}\n"
        f"  Completion tok : {total_completion_tokens}\n"
        f"  Total tokens   : {total_tokens}\n"
        f"  Footings found : {len(collected_footings)}\n"
        f"  ─────────────────────────────────────────"
    )

    if collected_footings:
        return json.dumps({"footings": collected_footings})

    print("  Tool extraction returned 0 footings -- falling back.")
    return extract_from_image(image_path, prompt_text)


def extract_from_image(image_path, prompt_text, retries=3):
    base64_image = encode_image(image_path)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            _image_content(base64_image),
                        ],
                    }
                ],
                temperature=0,
            )
            return clean_json_string(response.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"  Fallback extraction failed, retrying ({attempt}/{retries})...")
                time.sleep(5)
    raise RuntimeError(f"Extraction failed after {retries} retries: {last_error}")
