import os
import json

from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


def detect_pattern(pdf_path, temp_folder):
    """
    Detect footing schedule pattern (1 to 9)
    based STRICTLY on HEADER structure.
    """

    image_paths = convert_pdf_to_images(pdf_path, temp_folder)

    if not image_paths:
        raise Exception("No image generated for pattern detection.")

    first_image = image_paths[0]

    classification_prompt = """
You are an expert at identifying RCC FOOTING schedule header patterns.

Look ONLY at the HEADER structure.
Ignore all data rows completely.

There are EXACTLY 9 footing patterns.

Return ONLY one number:
1
2
3
4
5
6
7
8
9
10

No explanation.
No extra text.

=================================================
PATTERN 1
=================================================
Header contains:

FOOTING DETAILS
(MIX : M20)
SBC
Size
D TO d
Short Span
Long Span
COLUMN MARK (B)

=================================================
PATTERN 2
=================================================
Header contains:

FOOTING
PEDESTAL SIZE
CONC. MIX
HORI. STEEL
VERTI. STEEL
FOOTING SIZE
P.C.C. SIZE
DEPTH (d TO D)
REINF. //B
REINF. //L
FOOTING MIX
FLOOR
COL. MARK

Grid layout with bottom row labeled COL. MARK

=================================================
PATTERN 3
=================================================
Header contains:

FOOTING SCHEDULE
TOP STEEL
BOTTOM STEEL
DETAIL OF FOOTING
P.C.C.
COLUMNS MARKED
SHORT DIR
LONG DIR
SHORT DIR
LONG DIR
DEPTH (D)
SIZE
150 THK.

Key difference from Pattern 8:
- Pattern 3 uses SHORT DIR / LONG DIR (not SHORT SIDE / LONG SIDE)
- Pattern 3 has "DETAIL OF FOOTING" with DEPTH (D) and SIZE rows
- Pattern 3 has "COLUMNS MARKED" row (with AC/BC prefixed column IDs)
- Pattern 3 does NOT have "B x L", "t", or "STEEL // - B (SHORT SIDE) (BOTTOM FACE)"
If you see SHORT DIR, LONG DIR, COLUMNS MARKED, DETAIL OF FOOTING → it is Pattern 3, NOT Pattern 8.

=================================================
PATTERN 4
=================================================
Header contains:

FOOTING ID
SIZE OF FOOTING
LENGTH (L)
BREADTH (B)
DEPTH (D)
NOS.
GRID REFERENCE

=================================================
PATTERN 5
=================================================
Header contains:

SL NO.
TYPE
SIZE - L
SIZE - B
SIZE - D
TOP REINFORCEMENT - T1
TOP REINFORCEMENT - T2
BOTTOM REINFORCEMENT - B1
BOTTOM REINFORCEMENT - B2
NO. OF FOOTING


=================================================
PATTERN 6
=================================================
Header contains:

GROUP NO.
FOOTING NO.
EXCAVATION SIZE [LXB]
STEP-1 [LXB]
STEP-2 [LXB]
DEPTH - D
D1
D2
CON. MIX
FOOTING REIN BOTH WAYS
TOP-1
TOP-2
BOTTOM
FOOTING [L]
TYP. OF FOOTING

=================================================
PATTERN 7
=================================================
Header contains:

FOOTING DETAILS.
FOOTING NO.
P.C.C. (B' x L')
STEP-1 (B x L)
STEP-2 (B1 x L1)
STEP-1 (D-THK.)
STEP-2 (D-THK.)
STEP-1
TOP
STEEL // B
STEEL // L
BOTTOM
STEEL // B
STEEL // L
STEP-2
TOP
STEEL // B1
STEEL // L1
CONC. MIX

=================================================
PATTERN 8
=================================================
Header contains:

FOOTING SCHEDULE
B' x L' (PCC)
B x L
D
t
STEEL // - B (SHORT SIDE) (BOTTOM FACE)
STEEL // - L (LONG SIDE) (BOTTOM FACE)
STEEL // - B (SHORT SIDE) (TOP FACE)
STEEL // - L (LONG SIDE) (TOP FACE)
REMARK

Pattern 8 contains small 't' for thickness instead of capital 'T'.

Key difference from Pattern 3:
- Pattern 8 uses SHORT SIDE / LONG SIDE (not SHORT DIR / LONG DIR)
- Pattern 8 has "B x L" and "B' x L' (PCC)" rows (not "SIZE" and "DETAIL OF FOOTING")
- Pattern 8 has "STEEL // - B (SHORT SIDE)" format (not "TOP STEEL" / "BOTTOM STEEL")
- Pattern 8 does NOT have "COLUMNS MARKED" or "DETAIL OF FOOTING"
If you see SHORT DIR, LONG DIR, COLUMNS MARKED, DETAIL OF FOOTING → it is Pattern 3, NOT Pattern 8.

=================================================
PATTERN 9
=================================================
Header contains:

FOOTING SCHEDULE
RAFT-1
RAFT-2
RAFT-3
B' x L' (PCC)
B x L
D
T
STEEL // - B (SHORT SIDE) (BOTTOM FACE)
STEEL // - L (LONG SIDE) (BOTTOM FACE)
STEEL // - B (SHORT SIDE) (TOP FACE)
STEEL // - L (LONG SIDE) (TOP FACE)
REMARK

Pattern 9 contains capital 'T' for thickness and no 't' in the header.

=================================================
PATTERN 10
=================================================
Header contains:

FOOTING SCHEDULE
FOOTING NUMBERS
COLUMN NUMBERS
FOOTING TYPE
FOOTING DIMENSION - L
FOOTING DIMENSION - B
FOOTING DIMENSION - D
FOOTING REINFORCEMENT - BOTTOM
ALONG B 
ALONG L
FOOTING REINFORCEMENT - TOP
ALONG B 
ALONG L
SFR
REMARK

=================================================


Return ONLY the number.
"""

    result = extract_from_image(first_image, classification_prompt)
    result = result.strip()

    if not result.isdigit():
        raise Exception(f"Pattern detection failed. Model returned: {result}")

    return int(result)
