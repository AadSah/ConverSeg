REGION_DENSE_CAPTION_META_PROMPT = """
SYSTEM META-PROMPT: REGION-LEVEL DENSE CAPTION (ABSOLUTE REFERENCING)

MISSION
Given one image, **pick 5-7 high-value unique regions** and label them for natural conversational referencing.

PHASE 1 — REASONING (write first)
In <reasoning>, briefly answer:
- Scene type (portrait/indoor/outdoor/product/etc.)
- 5-7 most salient objects/areas
- Spatial layout (foreground/midground/background)
- For candidate regions, assess: referability, relational potential, disambiguation need, segmentation clarity
- Selection: which 5-7 regions you chose, why, and example prompts they enable

PHASE 2 — OUTPUT (strict format)
In <output>, list 5-7 lines:
- Indices 0..N-1, contiguous
- One per line: "[<index>: <label>]"
- <label> = <base_category> [distinctive_attributes] [coarse_location] [spatial_relation]
- ≤15 words per label; absolute, self-contained (no pronouns/anaphora)

LABELING RULES
- Base categories: common nouns (person, chair, laptop, window, tree, car, wall, floor, sky, etc.)
- Attributes: color/material/state/pattern/opacity when helpful (e.g., "white ceramic", "open", "transparent")
- Locations: top-left/top-center/top-right/left/center/right/bottom-left/bottom-center/bottom-right/foreground/midground/background
- Relations: "on/under/next to/behind/in front of/inside/beside <category>"
- Consistent terms; singular unless intentionally grouping similar items
- Parts: "<part> of <parent_category>" (parent must appear earlier)
- Stuff regions allowed if salient (sky, wall, floor, road, grass, water)
- Prefer clear, >50×50 px, well-bounded regions; skip tiny clutter (<24×24 px), heavy overlaps (>80%), ambiguous blobs
- Favor quality over quantity; in simple scenes, fewer regions are fine
- No invented objects or confidence statements

SELECTION STRATEGY
- Cover dominant subjects and interactive objects
- Ensure spatial and semantic diversity
- Choose regions that enable natural relations among them
- Use intentional groupings only when referenced as one (e.g., dual monitors)

FINAL RESPONSE FORMAT
<reasoning>
[concise analysis and selection rationale]
</reasoning>

<output>
[0: region description]
[1: region description]
...
</output>

EXAMPLE
<reasoning>
Indoor workspace; salient: person (foreground), laptop, cup, wooden desk, window (background).
Chosen for referability and relations (on desk, behind person). Prompts: "highlight the laptop", "segment the cup next to laptop".
</reasoning>

<output>
[0: person, blue shirt, center foreground]
[1: laptop, silver, on desk]
[2: cup, white ceramic, right side of desk]
[3: desk, wooden surface, midground]
[4: window, glass panes, background behind person]
</output>
"""