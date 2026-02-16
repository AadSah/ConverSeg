ENTITIES_META_PROMPT = """
You are an expert AI tasked with generating difficult, abstract segmentation prompts. You will be given an image along with another copy of it where available segmentation masks are overlaid and numbered directly on the regions, and a dense caption (a numbered list of available segmentation masks).

**TASK:** Design up to 3 challenging segmentation prompts. Each prompt must cleverly distinguish a subset of the available regions based on a shared visual property that requires careful inspection of the image.

---

### GUIDING PRINCIPLE: The Contextual Plausibility Rule

This is the most important rule. The prompt must be logical and relevant for the **entire image scene**, not just for the limited set of masked regions. It should not create nonsensical scenarios. **The prompt you generate MUST apply *only* to the desired masked region(s) and to NO other un-masked regions in the image.**

*   **THE PROBLEM TO AVOID:** Generating a prompt where a better answer exists in an un-masked part of the image.
*   **BAD EXAMPLE:** The image contains a `table` (unmasked) and a `chair` (masked). A prompt like "Segment a surface to place a drink on" which returns the `chair` is **INVALID**. While one *could* place a drink on a chair, the unmasked table is the far superior and intended answer, making the prompt misleading and contextually poor.
*   **GOOD EXAMPLE:** In the same scene, if the chair is wooden and the table is metal, a prompt like "Segment the wooden furniture" is **VALID**. This prompt targets a specific *visual property* (material) that correctly and unambiguously applies to the masked region without creating a conflict with unmasked objects.

**Your primary goal is to focus on visual properties (materials, parts, states, textures, groups) that genuinely differentiate the target regions, rather than ambiguous functional affordances.**

---

### CRITICAL CONSTRAINTS

**1. Abstract & Non-Trivial Selection:**
   - The prompt MUST require visual discrimination between multiple candidate regions. It should not be solvable by just reading the caption.
   - **VALID:** "Segment the people wearing hats" (requires checking each person).
   - **INVALID:** "Segment the car" (if there's only one car, it's a simple lookup).
   - **TEST:** Does the user have to visually compare multiple regions to find the answer?

**2. The Proper Subset Rule:**
   - You must define a list of `candidates`—plausible regions from the caption that a user might consider.
   - You must define a `satisfying` list—the regions from the `candidates` that *actually* meet the prompt's criteria after visual inspection.
   - The `satisfying` set MUST be a **strict subset** of the `candidates` set (i.e., `satisfying` cannot be identical to `candidates`). An empty `satisfying` set is valid.

**3. Formatting & Output:**
   - **Prompt:** Maximum 10 words, starting with an actionable verb (e.g., "Segment...", "Identify...").
   - **Concept Families:** Choose up to 3 diverse concept families from the list below.
   - **JSON Only:** The final output must be ONLY the JSON object, with no commentary or markdown fences.

---

### CONCEPT FAMILIES

1.  **instances_and_agents:** Multiple instances of same category; agents performing actions.
2.  **stuff_surfaces_regions:** Amorphous materials (sky, grass); extended surfaces; spatial zones.
3.  **object_parts_and_functional_regions:** Subparts (handles, wheels); contact areas; interaction zones.
4.  **materials_and_optics:** Material properties: transparent, reflective, textured, etc.
5.  **articulated_and_states:** State variations: open/closed, on/off, full/empty, clean/dirty.
6.  **groups_and_piles:** Collections, clusters, or stacks of similar items.
7.  **anticipatory_entity_state:** Vulnerability assessment: likely to spoil/fall/break first.
8.  **unusual_entity:** Atypical appearance, uncommon variants, anomalous instances.

---

### REASONING PROCESS

**Step 1: Holistic Analysis & Concept Generation.**
   - First, analyze the **entire image** to understand the scene, objects, and their relationships.
   - Next, review the **available masks** in the dense caption.
   - Identify visual properties (e.g., "all the rusty metal items," "the containers that are open") that create an interesting subgroup *within the available masks*. Ask: "What shared visual trait makes regions [X, Y] special compared to other available regions like [A, B, C]?"

**Step 2: Prompt Authoring & Sanity Check.**
   - Author a concise prompt based on the concept from Step 1.
   - Define the `candidates` and `satisfying` lists.
   - **Perform the Sanity Check:** Does this prompt pass the **Contextual Plausibility Rule**? Is there an obvious, un-masked object in the scene that makes my prompt nonsensical or misleading? If so, discard or refine the prompt to be more specific to a visual property.

---

**Dense caption:**
{DENSE_CAPTION}

**Return ONLY the JSON object.**
"""


SPATIAL_META_PROMPT = """
You are an expert AI tasked with generating difficult, abstract segmentation prompts about spatial relationships and layouts. You will be given an image along with another copy of it where available segmentation masks are overlaid and numbered directly on the regions, and a dense caption (a numbered list of available segmentation masks).

**TASK:** Design up to 3 challenging segmentation prompts. Each prompt must require visual inspection to determine how entities are arranged and positioned relative to each other or the viewer.

---

### GUIDING PRINCIPLE: The Contextual Plausibility Rule for Spatial Relations

This is the most important rule. The spatial property you prompt for must be logical and plausible for the **entire image scene**, especially for superlative claims (e.g., nearest, largest, leftmost). **The prompt you generate MUST apply *only* to the desired masked region(s) and to NO other un-masked regions in the image.**

*   **THE PROBLEM TO AVOID:** Generating a prompt (e.g., "the nearest object") where the correct answer is an un-masked object, making the prompt's selection of a masked object misleading.
*   **BAD EXAMPLE:** The image shows a masked `cup` and an unmasked `table leg` that is clearly much closer to the camera. A prompt like "Segment the nearest object" which returns the `cup` is **INVALID**. The unmasked table leg is the true answer, making the prompt's result incorrect.
*   **GOOD EXAMPLE:** In the same scene, with multiple masked cups, a prompt like "Segment cups on the left side of the table" is **VALID**. This focuses on a verifiable relationship between masked items and a reference surface without making a false global, superlative claim. Another valid prompt could be "Segment the leftmost cup," as it correctly ranks *within the set of available cups*.

**Your primary goal is to focus on verifiable spatial relationships, and if using superlatives, ensure they are appropriately constrained (e.g., "leftmost *cup*") to avoid conflicts with un-masked objects.**

---

### CRITICAL CONSTRAINTS

**1. Abstract & Non-Trivial Selection:**
   - The prompt MUST require visual discrimination between multiple candidate regions to determine their spatial relationship.
   - **VALID:** "Segment the items on the top shelf" (requires checking multiple items and shelves).
   - **INVALID:** "Segment the item on the left" (if there is only one item on the left).
   - **TEST:** Does the user have to visually analyze position, distance, orientation, or containment to find the answer?

**2. The Proper Subset Rule:**
   - You must define a list of `candidates`—plausible regions from the caption that a user might consider for the spatial property.
   - You must define a `satisfying` list—the regions from the `candidates` that *actually* exhibit the property after visual inspection.
   - The `satisfying` set MUST be a **strict subset** of the `candidates` set (i.e., `satisfying` cannot be identical to `candidates`). An empty `satisfying` set is valid.

**3. Formatting & Output:**
   - **Prompt:** Maximum 10 words, starting with an actionable verb (e.g., "Segment...", "Identify...").
   - **Concept Families:** Choose up to 3 diverse concept families from the list below.
   - **JSON Only:** The final output must be ONLY the JSON object, with no commentary or markdown fences.

---

### CONCEPT FAMILIES

1.  **topological:** Containment and contact (in/on/under/touching/adjacent).
2.  **egocentric_viewer:** Viewer-centric directions (left/right/front/behind).
3.  **allocentric_object:** Object-relative directions (left/right of object X).
4.  **metric_bins:** Distance and orientation (within_reach, near/far, tilted/angled).
5.  **support_contact:** Physical support (surfaces supporting stacks, floor contact).
6.  **visibility_occlusion:** Visibility state (partially occluded, fully visible).
7.  **counting_ordering:** Ranked selection (k_largest, smallest, k_leftmost, tallest).
8.  **unusual_location:** Spatial anomalies (unusually placed, out-of-place items).

---

### REASONING PROCESS

**Step 1: Holistic Spatial Analysis.**
   - First, analyze the **entire image** to understand the layout. Where are things? What is on what? What is near what?
   - Next, review the **available masks** in the dense caption.
   - Identify a spatial relationship (e.g., "items inside containers," "objects touching the wall," "the three tallest chairs") that creates an interesting subgroup *within the available masks*.

**Step 2: Prompt Authoring & Sanity Check.**
   - Author a concise prompt based on the spatial relationship from Step 1.
   - Define the `candidates` and `satisfying` lists.
   - **Perform the Sanity Check:** Does this prompt pass the **Contextual Plausibility Rule**? If I use a superlative like 'nearest' or 'largest', is there a more obvious, un-masked object that fits the description better? If so, either discard the prompt or constrain it (e.g., from "the largest object" to "the largest piece of furniture").

---

**Dense caption:**
{DENSE_CAPTION}

**Return ONLY the JSON object.**
"""


RELATIONS_META_PROMPT = """
You are an expert AI tasked with generating difficult, abstract segmentation prompts about actions, states, and relationships. You will be given an image along with another copy of it where available segmentation masks are overlaid and numbered directly on the regions, and a dense caption (a numbered list of available segmentation masks).

**TASK:** Design up to 3 challenging segmentation prompts. Each prompt must require visual inspection to determine how entities are interacting, what state they are in, or what role they play in an event.

---

### GUIDING PRINCIPLE: The Contextual Plausibility Rule for Relations

This is the most important rule. The relationship (action, state, role) you prompt for must be logical and plausible within the context of the **entire image scene**, not just for the limited set of masked regions. **The prompt you generate MUST apply *only* to the desired masked region(s) and to NO other un-masked regions in the image.**

*   **THE PROBLEM TO AVOID:** Identifying a participant in an action (e.g., the instrument or the target) that is a less plausible choice than an obvious, un-masked entity in the scene.
*   **BAD EXAMPLE:** The image shows a person (masked) holding a `pen` (masked) near a large, unmasked `whiteboard`. A prompt like "Segment the target of the writing action" that returns the person's other `hand` (also masked) is **INVALID**. The whiteboard is the clear, intended target, making the prompt's answer nonsensical.
*   **GOOD EXAMPLE:** In the same scene, a prompt like "Segment the object being held by the person" which returns the `pen` is **VALID**. This focuses on a direct, unambiguous physical relationship (holding) that is undeniably true and doesn't conflict with the broader scene's inferred actions.

**Your primary goal is to anchor prompts in visually verifiable interactions and states, avoiding high-level intentions that could be contradicted by more obvious, un-masked elements.**

---

### CRITICAL CONSTRAINTS

**1. Abstract & Non-Trivial Selection:**
   - The prompt MUST require visual discrimination between multiple candidate regions to determine their role, state, or action.
   - **VALID:** "Segment the people who are holding something" (requires checking each person).
   - **INVALID:** "Segment the person sitting" (if there is only one person and their state is obvious).
   - **TEST:** Does the user have to visually analyze poses, positions, and interactions to find the answer?

**2. The Proper Subset Rule:**
   - You must define a list of `candidates`—plausible regions from the caption that a user might consider for the relationship.
   - You must define a `satisfying` list—the regions from the `candidates` that *actually* fit the relational criteria after visual inspection.
   - The `satisfying` set MUST be a **strict subset** of the `candidates` set (i.e., `satisfying` cannot be identical to `candidates`). An empty `satisfying` set is valid.

**3. Formatting & Output:**
   - **Prompt:** Maximum 10 words, starting with an actionable verb (e.g., "Segment...", "Identify...").
   - **Concept Families:** Choose up to 3 diverse concept families from the list below.
   - **JSON Only:** The final output must be ONLY the JSON object, with no commentary or markdown fences.

---

### CONCEPT FAMILIES

1.  **atomic_posture:** Physical actions and postures (holding, wearing, carrying, grasping, sitting).
2.  **event_state:** Temporal and state markers (before/during/after action; cut/uncut; open/closed).
3.  **semantic_role:** Action participants (agent, instrument, patient/target).
4.  **anticipation:** Predictive cues (about-to-be targets, likely spill destination, imminent contact).

---

### REASONING PROCESS

**Step 1: Holistic Relational Analysis.**
   - First, analyze the **entire image** to understand the scene's dynamics. What is happening? Who is doing what to what? What states are objects in?
   - Next, review the **available masks** in the dense caption.
   - Identify a relationship (e.g., "items being worn," "tools currently in use," "food that has been sliced") that creates an interesting subgroup *within the available masks*.

**Step 2: Prompt Authoring & Sanity Check.**
   - Author a concise prompt based on the relationship concept from Step 1.
   - Define the `candidates` and `satisfying` lists.
   - **Perform the Sanity Check:** Does this prompt pass the **Contextual Plausibility Rule**? Is there a more obvious, un-masked participant in this action that makes my answer illogical? If so, discard the prompt or refine it to focus on a more direct, observable interaction.

---

**Dense caption:**
{DENSE_CAPTION}

**Return ONLY the JSON object.**
"""

AFFORDANCES_META_PROMPT = """
You are an expert AI tasked with generating difficult, abstract segmentation prompts about functional affordances. You will be given an image along with another copy of it where available segmentation masks are overlaid and numbered directly on the regions, and a dense caption (a numbered list of available segmentation masks).

**TASK:** Design up to 3 challenging segmentation prompts. Each prompt must require visual inspection to determine an object's functional properties, usability, or potential uses based on the current scene context.

---

### GUIDING PRINCIPLE: The Contextual Plausibility Rule for Affordances

This is the most important rule. The affordance you prompt for must be logical and plausible for the **entire image scene**, not just for the limited set of masked regions. It must not lead to a nonsensical or clearly suboptimal choice. **The prompt you generate MUST apply *only* to the desired masked region(s) and to NO other un-masked regions in the image.**

*   **THE PROBLEM TO AVOID:** Promoting a masked object for a specific function when a better, more obvious, un-masked object for that same function is clearly visible.
*   **BAD EXAMPLE:** The image contains a large, unmasked `dining table` and a small, masked `stool`. A prompt like "Segment a surface to place a laptop on" which returns the `stool` is **INVALID**. The unmasked table is the primary and far more appropriate surface, making the prompt misleading.
*   **GOOD EXAMPLE:** In the same scene, a prompt like "Segment a portable seat" which returns the `stool` is **VALID**. This targets a more specific functional property (portability) that correctly and uniquely applies to the masked stool without creating a conflict with the unmasked table.

**Your primary goal is to identify unique functional conditions, states, or properties that genuinely apply to the masked regions without creating these logical conflicts.**

---

### CRITICAL CONSTRAINTS

**1. Abstract & Non-Trivial Selection:**
   - The prompt MUST require visual discrimination between multiple candidate regions. It should not be solvable by just reading the caption.
   - **VALID:** "Segment chairs that are currently sittable" (requires checking each chair for obstructions).
   - **INVALID:** "Segment the sink" (if there's only one, it's a simple lookup).
   - **TEST:** Does the user have to visually assess the state, position, or condition of multiple items to find the answer?

**2. The Proper Subset Rule:**
   - You must define a list of `candidates`—plausible regions from the caption that a user might consider for the affordance.
   - You must define a `satisfying` list—the regions from the `candidates` that *actually* provide the affordance after visual inspection.
   - The `satisfying` set MUST be a **strict subset** of the `candidates` set (i.e., `satisfying` cannot be identical to `candidates`). An empty `satisfying` set is valid.

**3. Formatting & Output:**
   - **Prompt:** Maximum 10 words, starting with an actionable verb (e.g., "Segment...", "Identify...").
   - **Concept Families:** Choose up to 3 diverse concept families from the list below.
   - **JSON Only:** The final output must be ONLY the JSON object, with no commentary or markdown fences.

---

### CONCEPT FAMILIES

1.  **context_dependent:** Usability given current scene conditions (walkable_now, sittable_now, reachable_now).
2.  **context_independent:** Canonical function regardless of state (water_source_canonical, seating_canonical).
3.  **negative_affordance:** Inappropriate uses (not_for_liquids, not_safe_to_touch, not_for_sitting).
4.  **counterfactual_affordance:** Creative/improvised uses (could_prop_door, could_be_step_stool).
5.  **state_dependent_and_agent_conditional:** Requires a specific state or agent (openable_not_blocked, operable_by_child).
6.  **anticipatory_affordance:** Soon-to-emerge function (will_soon_be_hot, about_to_be_ready).

---

### REASONING PROCESS

**Step 1: Holistic Affordance Analysis.**
   - First, analyze the **entire image** to understand the scene and the potential actions. What can be done here? What are things for? What is currently usable versus blocked or unsafe?
   - Next, review the **available masks** in the dense caption.
   - Identify a functional property (e.g., "unobstructed seating," "surfaces safe for hot items") that creates an interesting subgroup *within the available masks*.

**Step 2: Prompt Authoring & Sanity Check.**
   - Author a concise prompt based on the affordance concept from Step 1.
   - Define the `candidates` and `satisfying` lists.
   - **Perform the Sanity Check:** Does this prompt pass the **Contextual Plausibility Rule**? Is there a better, un-masked object for this exact function? If so, discard the prompt or make it more specific by adding a constraint (e.g., material, state, portability) that makes the answer unique and logical.

---

**Dense caption:**
{DENSE_CAPTION}

**Return ONLY the JSON object.**
"""

PHYSICS_META_PROMPT = """
You are an expert AI tasked with generating difficult, abstract segmentation prompts about physics, stability, and safety. You will be given an image along with another copy of it where available segmentation masks are overlaid and numbered directly on the regions, and a dense caption (a numbered list of available segmentation masks).

**TASK:** Design up to 3 challenging segmentation prompts. Each prompt must require visual inspection and physical reasoning to assess an object's stability, containment, potential hazards, or other physical properties.

---

### GUIDING PRINCIPLE: The Contextual Plausibility Rule for Physics

This is the most important rule. The physical property you prompt for must be logical and plausible for the **entire image scene**, not just for the limited set of masked regions. Avoid making superlative claims that are contradicted by un-masked objects. **The prompt you generate MUST apply *only* to the desired masked region(s) and to NO other un-masked regions in the image.**

*   **THE PROBLEM TO AVOID:** Generating a prompt that identifies a masked object with a specific physical property (e.g., "most unstable") when a far better example exists in an un-masked part of the image.
*   **BAD EXAMPLE:** The image contains a slightly tilted `book` (masked) and a large, unmasked `vase` teetering on the very edge of a table. A prompt like "Segment the object most likely to fall" that returns the `book` is **INVALID**. The unmasked vase is the far more obvious and correct answer, making the prompt's result misleading.
*   **GOOD EXAMPLE:** In the same scene, with multiple masked books, a prompt like "Segment books not fully supported by the surface" is **VALID**. This focuses on a specific, verifiable condition of the masked items without making a global, superlative claim that is contradicted by the unmasked vase.

**Your primary goal is to focus on verifiable physical states (instability, spill risk, sharpness) of masked objects, rather than making global claims that fail the broader scene context.**

---

### CRITICAL CONSTRAINTS

**1. Abstract & Non-Trivial Selection:**
   - The prompt MUST require visual discrimination between multiple candidate regions to assess their physical properties.
   - **VALID:** "Segment containers at risk of spilling" (requires checking the fill level and stability of each container).
   - **INVALID:** "Segment the sharp knife" (if there is only one knife and its sharpness is a default property).
   - **TEST:** Does the user have to visually analyze geometry, support, material, and position to find the answer?

**2. The Proper Subset Rule:**
   - You must define a list of `candidates`—plausible regions from the caption that a user might consider for the physical property.
   - You must define a `satisfying` list—the regions from the `candidates` that *actually* exhibit the property after visual inspection.
   - The `satisfying` set MUST be a **strict subset** of the `candidates` set (i.e., `satisfying` cannot be identical to `candidates`). An empty `satisfying` set is valid.

**3. Formatting & Output:**
   - **Prompt:** Maximum 10 words, starting with an actionable verb (e.g., "Segment...", "Identify...").
   - **Concept Families:** Choose up to 3 diverse concept families from the list below.
   - **JSON Only:** The final output must be ONLY the JSON object, with no commentary or markdown fences.

---

### CONCEPT FAMILIES

1.  **support_stability:** Structural stability (likely_to_fall, safe_to_stand_on, likely_to_tip, fragile_stack_part).
2.  **containment_and_flow:** Liquid/material containment (risk_of_spilling, lacks_lid, overfilled).
3.  **collision_obstruction_passability:** Path blocking (blocking_entry, obstructing_route).
4.  **hazard:** Safety risks (requires_protection_to_touch, slippery_surface, sharp_edge).
5.  **counterfactuals:** What-if scenarios (would_collide_if_opened, would_collapse_if_removed).
6.  **unusual:** Physical anomalies (physically_implausible, unsupported_floating, defying_gravity).

---

### REASONING PROCESS

**Step 1: Holistic Physical Analysis.**
   - First, analyze the **entire image** for physical properties. What looks unstable, hazardous, or blocked? What are the risks?
   - Next, review the **available masks** in the dense caption.
   - Identify a physical property (e.g., "precarious balance," "potential spill source," "sharp edges") that creates an interesting subgroup *within the available masks*.

**Step 2: Prompt Authoring & Sanity Check.**
   - Author a concise prompt based on the physical property from Step 1.
   - Define the `candidates` and `satisfying` lists.
   - **Perform the Sanity Check:** Does this prompt pass the **Contextual Plausibility Rule**? If I ask for the "most" of something, is there an un-masked object that is a better example? If so, discard or rephrase the prompt to be more specific and less superlative (e.g., change "most unstable" to "unstable items").

---

**Dense caption:**
{DENSE_CAPTION}

**Return ONLY the JSON object.**
"""