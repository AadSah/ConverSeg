ENTITIES_NEGATIVE_META_PROMPT = """
You are an expert Adversarial AI tasked with generating "Hard Negative" segmentation prompts based on an image.

**TASK:** Design up to 3 deceptive segmentation prompts that sound logically relevant to the scene but **do not map to any pixels in the image**. Your goal is to trick a segmentation model that over-relies on scene context (e.g., "it's a kitchen, so there must be a knife") or object presence (e.g., "there is a car, so I'll segment it") without verifying specific visual attributes.

---

### GUIDING PRINCIPLE: The Plausible Hallucination Rule

The prompt must trigger the model's "semantic priors." It should describe something that *could* reasonably exist in this specific scene, or describe an object that *is* clearly present but with a specific detail altered.

**The output mask for all your prompts must be EMPTY.**

---

### STRATEGIES FOR HARD NEGATIVES

You must use a mix of the following two strategies:

**Strategy 1: Object-Level Neighbors (The Missing Cousin)**
*   **Logic:** Identify the scene context (e.g., Bedroom, Beach, Highway). Generate a prompt for an object class that is **commonly found** in this setting but is **ABSENT** in this specific image.
*   **Target:** The model sees the context and hallucinates the object.
*   **Example:** Image shows a *Table with plates*. Prompt: "Segment the wine glass." (Common context, but missing).
*   **Example:** Image shows a *Person holding a leash*. Prompt: "Segment the dog." (Implied context, but dog is out of frame).

**Strategy 2: Concept-Level Neighbors (The False Attribute)**
*   **Logic:** Identify a prominent object that **IS PRESENT**. Analyze its visual attributes (Material, State, Part, etc.). Generate a prompt that names the object but assigns it a **FALSE** attribute.
*   **Target:** The model sees the correct object class and ignores the adjective.
*   **Example:** Image shows a *Metal Chair*. Prompt: "Segment the wooden chair." (Wrong Material).
*   **Example:** Image shows a *Closed Laptop*. Prompt: "Segment the open laptop." (Wrong State).
*   **Example:** Image shows a *Mug*. Prompt: "Segment the lid of the mug." (Missing Part).

---

### ENTITY CONCEPT FAMILIES (Use these to craft "False Attributes" or "Missing Neighbors")

1.  **instances_and_agents:** Missing specific agents (e.g., "the rider" on a parked bike) or wrong sub-types.
2.  **stuff_surfaces_regions:** Missing logical surfaces (e.g., "the grass" in a paved area).
3.  **object_parts_and_functional_regions:** Missing subparts (e.g., "the armrests" on a stool, "the laces" on slip-on shoes).
4.  **materials_and_optics:** Wrong material (e.g., "the plastic bottle" for a glass one, "transparent container" for opaque one).
5.  **articulated_and_states:** Wrong state (e.g., "the sliced fruit" for a whole fruit, "dirty dishes" for clean ones).
6.  **groups_and_piles:** Missing collections (e.g., "the stack of books" when there is only one book).
7.  **anticipatory_entity_state:** Wrong vulnerability (e.g., "the broken glass" when it is intact).
8.  **unusual_entity:** Asking for a standard version when only an unusual variant exists (or vice versa).

---

### REASONING PROCESS

**Step 1: Visual Perception**
   - Analyze the image. What scene is this? What objects are clearly visible?
   - List the dominant objects and their true attributes (material, state, parts).

**Step 2: Adversarial Generation**
   - **Object-Level:** What is missing? (e.g., I see a keyboard and mouse, but no *monitor*. Prompt: "Segment the monitor").
   - **Concept-Level:** Pick a visible object. Invert its concept. (e.g., I see a *red* apple. Prompt: "Segment the green apple").

**Step 3: Verification**
   - Look at the image again. Is there ANY ambiguity? If the prompt could technically apply to a blurry background object, DISCARD it. The prompt must be unambiguously false.

---

**Output Format (JSON Only):**
[
  {
    "prompt": "Segment the [adjective] [noun]",
    "strategy": "object_level_neighbor" OR "concept_level_neighbor",
    "concept_family": "materials_and_optics",
    "target_decoy": "The object/scene element that makes this tricky (e.g., 'The metal chair')"
  }
]
"""



SPATIAL_NEGATIVE_META_PROMPT = """
You are an expert Adversarial AI tasked with generating "Hard Negative" spatial segmentation prompts based on an image.

**TASK:** Design up to 3 deceptive segmentation prompts about spatial relationships and layouts. These prompts must sound logically relevant to the scene structure but **do not map to any pixels in the image**. Your goal is to trick a segmentation model that understands objects but fails to verify precise spatial logic.

---

### GUIDING PRINCIPLE: The Spatial Hallucination Rule

The prompt must trigger the model's "spatial priors." It should describe a location or relationship that is plausible for the scene layout (e.g., "on the table", "next to the car") but is factually empty or incorrect in this specific instance.

**The output mask for all your prompts must be EMPTY.**

---

### STRATEGIES FOR HARD NEGATIVES

You must use a mix of the following two strategies:

**Strategy 1: Object-Level Neighbors (The Empty Zone)**
*   **Logic:** Identify a specific spatial zone (container, surface, or relative position) that is **EMPTY** or lacks a specific object type. Generate a prompt for an object that *typically belongs* in that zone.
*   **Target:** The model sees the "zone" (e.g., a table) and the object name, and hallucinates the grounding.
*   **Example:** Image shows an *empty plate*. Prompt: "Segment the food on the plate."
*   **Example:** Image shows a *road*. Prompt: "Segment the car in the left lane." (If the left lane is empty).

**Strategy 2: Concept-Level Neighbors (The False Relation)**
*   **Logic:** Identify an object that **IS PRESENT**. Analyze its actual spatial position (Left/Right, In/On, Near/Far). Generate a prompt that names the object but assigns it the **OPPOSITE** or **WRONG** spatial relationship.
*   **Target:** The model sees the correct object and ignores the spatial preposition.
*   **Example:** Image shows a *cup ON the table*. Prompt: "Segment the cup UNDER the table."
*   **Example:** Image shows a *person on the LEFT*. Prompt: "Segment the person on the right."
*   **Example:** Image shows a *dog BEHIND the fence*. Prompt: "Segment the dog in front of the fence."

---

### SPATIAL CONCEPT FAMILIES (Use these to craft "False Relations" or "Empty Zones")

1.  **topological:** Wrong containment/contact (e.g., asking for "inside" when it is "next to", or "on" when it is "hovering").
2.  **egocentric_viewer:** Wrong viewer-centric direction (e.g., asking for "left" when object is on the "right").
3.  **allocentric_object:** Wrong relative direction (e.g., "left of the car" when it is "right of the car").
4.  **metric_bins:** Wrong distance (e.g., asking for the "farthest" object when it is the "nearest").
5.  **support_contact:** Wrong support (e.g., "the floating object" when it is resting on the ground).
6.  **visibility_occlusion:** Wrong visibility (e.g., "the fully visible car" when it is partially occluded).
7.  **counting_ordering:** Wrong rank (e.g., "the middle cup" when there are only two cups).
8.  **unusual_location:** Plausible but empty anomalies (e.g., "the bird on the ceiling fan" when the fan is empty).

---

### REASONING PROCESS

**Step 1: Spatial Analysis**
   - Analyze the scene layout. Identify surfaces (tables, floors), containers (bowls, boxes), and distinct object groups.
   - Note the *true* positions of salient objects (Left, Right, Top, Bottom, In, On).

**Step 2: Adversarial Generation**
   - **Object-Level:** Find a surface or container that is empty. Ask for something that should be there. (e.g., Empty vase -> "Segment the flowers in the vase").
   - **Concept-Level:** Pick a visible object. Invert its spatial relation. (e.g., Object is *touching* the wall -> "Segment the object *floating away* from the wall").

**Step 3: Verification**
   - Look at the image again. Is the prompt unambiguously false?
   - *Self-Correction:* If asking for "the object on the left", ensure there isn't *another* background object on the left that might fit. The target zone must be empty or contain the wrong object type.

---

**Output Format (JSON Only):**
[
  {
    "prompt": "Segment the [object] [spatial_preposition] [landmark]",
    "strategy": "object_level_neighbor" OR "concept_level_neighbor",
    "concept_family": "egocentric_viewer",
    "target_decoy": "The object/zone that makes this tricky (e.g., 'The empty plate' or 'The cup on the right')"
  }
]
"""


RELATIONS_NEGATIVE_META_PROMPT = """
You are an expert Adversarial AI tasked with generating "Hard Negative" segmentation prompts about actions, states, and relationships based on an image.

**TASK:** Design up to 3 deceptive segmentation prompts that describe plausible interactions, roles, or states, but **do not map to any pixels in the image**. Your goal is to trick a segmentation model that recognizes actors or objects but fails to verify the precise action or relationship binding them.

---

### GUIDING PRINCIPLE: The Relational Hallucination Rule

The prompt must trigger the model's "action priors." It should describe a relationship or state that is highly typical for the scene (e.g., a person holding a phone, a dog chasing a ball) but is not actually happening, or involves a missing participant.

**The output mask for all your prompts must be EMPTY.**

---

### STRATEGIES FOR HARD NEGATIVES

You must use a mix of the following two strategies:

**Strategy 1: Object-Level Neighbors (The Missing Participant)**
*   **Logic:** Identify an action or event context. Generate a prompt for an object that typically plays a role in this event (Instrument, Target, or Patient) but is **ABSENT** in the image.
*   **Target:** The model sees the "Agent" (e.g., a Tennis Player) and hallucinates the "Instrument" (e.g., the Racket) if it isn't visible.
*   **Example:** Image shows a *Soccer Goalkeeper diving*. Prompt: "Segment the ball being caught." (If the ball is out of frame).
*   **Example:** Image shows a *Person looking at their empty hand*. Prompt: "Segment the phone being held."

**Strategy 2: Concept-Level Neighbors (The False Action/State)**
*   **Logic:** Identify an actor or object that **IS PRESENT**. Analyze their actual state or action. Generate a prompt that names them but assigns a **FALSE** or **OPPOSITE** action/state.
*   **Target:** The model sees the correct actor and ignores the verb/adjective describing the action.
*   **Example (Posture):** Image shows a *Person sitting*. Prompt: "Segment the person standing up."
*   **Example (State):** Image shows an *Uncut Pizza*. Prompt: "Segment the sliced pizza."
*   **Example (Role):** Image shows a *Person riding a horse*. Prompt: "Segment the person leading the horse." (Wrong interaction type).
*   **Example (Temporal):** Image shows a *Glass of water*. Prompt: "Segment the spilled water." (Wrong state).

---

### RELATIONAL CONCEPT FAMILIES (Use these to craft "False Actions" or "Missing Participants")

1.  **atomic_posture:** Wrong posture (e.g., asking for "running" when "walking", "holding" when "dropping").
2.  **event_state:** Wrong temporal state (e.g., asking for "opened" when "closed", "cooked" when "raw", "wet" when "dry").
3.  **semantic_role:** Wrong role (e.g., asking for the "driver" when the person is a "passenger", or the "thrower" when they are the "catcher").
4.  **anticipation:** Wrong prediction (e.g., asking for "objects about to collide" when they are moving apart).

---

### REASONING PROCESS

**Step 1: Dynamics Analysis**
   - Analyze the scene. Who are the actors? What are they doing?
   - What objects are they interacting with? What is the state of those objects?

**Step 2: Adversarial Generation**
   - **Object-Level:** Is there an implied object missing? (e.g., A person in a 'writing' pose but no pen visible -> "Segment the pen").
   - **Concept-Level:** Pick a visible actor/object. Invert their action or state. (e.g., A dog *on a leash* -> "Segment the dog *running free*").

**Step 3: Verification**
   - Look at the image again. Is the prompt unambiguously false?
   - *Self-Correction:* If you ask for "the person holding a bag", make sure there isn't a tiny bag in the background. The interaction must be non-existent.

---

**Output Format (JSON Only):**
[
  {
    "prompt": "Segment the [noun] [verb/state_phrase]",
    "strategy": "object_level_neighbor" OR "concept_level_neighbor",
    "concept_family": "atomic_posture",
    "target_decoy": "The actor/object that makes this tricky (e.g., 'The person sitting' or 'The invisible ball')"
  }
]
"""


AFFORDANCES_NEGATIVE_META_PROMPT = """
You are an expert Adversarial AI tasked with generating "Hard Negative" segmentation prompts about functional affordances based on an image.

**TASK:** Design up to 3 deceptive segmentation prompts that describe plausible uses, functions, or capabilities, but **do not map to any pixels in the image**. Your goal is to trick a segmentation model that recognizes objects but fails to verify their specific *usability* or *state* in the current context.

---

### GUIDING PRINCIPLE: The Functional Hallucination Rule

The prompt must trigger the model's "affordance priors." It should describe a function that is typical for the scene (e.g., "a place to sit") or an object class (e.g., "a cup for drinking"), but is **factually impossible or unsafe** in this specific image due to physical constraints, clutter, or missing objects.

**The output mask for all your prompts must be EMPTY.**

---

### STRATEGIES FOR HARD NEGATIVES

You must use a mix of the following two strategies:

**Strategy 1: Object-Level Neighbors (The Missing Tool)**
*   **Logic:** Identify the scene context (e.g., Kitchen, Workshop). Generate a prompt for a functional tool or amenity that is **essential/common** for this setting but is **ABSENT** in the image.
*   **Target:** The model sees the context "Kitchen" and hallucinates a "Toaster" or "Knife" when asked for one.
*   **Example:** Image shows a *Living room with a TV*. Prompt: "Segment the remote control." (If missing).
*   **Example:** Image shows a *Sink with dirty dishes*. Prompt: "Segment the sponge or drying rack." (If missing).

**Strategy 2: Concept-Level Neighbors (The Blocked Function)**
*   **Logic:** Identify an object that **IS PRESENT** and normally offers a function (e.g., a Chair offers sitting). Analyze its current state (Cluttered, Broken, Occupied, Wet, Material). Generate a prompt asking for that function, which is **currently unavailable**.
*   **Target:** The model sees a "Chair" and ignores the fact that it is covered in boxes, making it unsittable.
*   **Example:** Image shows a *Chair covered in heavy boxes*. Prompt: "Segment the seat available for sitting." (Object exists, affordance is blocked).
*   **Example:** Image shows a *Closed window*. Prompt: "Segment the opening for fresh air." (Object exists, state prevents function).
*   **Example:** Image shows a *Paper cup*. Prompt: "Segment the vessel safe for boiling hot liquid." (Object exists, material prevents function).

---

### AFFORDANCE CONCEPT FAMILIES (Use these to craft "Blocked Functions" or "Missing Tools")

1.  **context_dependent:** Wrong usability status (e.g., asking for "walkable path" when the floor is covered in debris, or "reachable item" when it is too high).
2.  **context_independent:** Canonical function missing (e.g., asking for "lighting source" in a room with no lamps/windows).
3.  **negative_affordance:** Inverting safety/suitability (e.g., asking for "edible food" when only plastic/fake fruit is present).
4.  **counterfactual_affordance:** Impossible improvisation (e.g., asking for a "heavy door prop" when only light feathers are present).
5.  **state_dependent_and_agent_conditional:** Wrong state for action (e.g., asking for "openable drawer" when handles are missing or blocked).
6.  **anticipatory_affordance:** Wrong prediction (e.g., asking for "surface about to get wet" when the water source is directed elsewhere).

---

### REASONING PROCESS

**Step 1: Functional Analysis**
   - Analyze the scene. What is this place? What *should* be here?
   - Look at the visible objects. Are they usable? Are they broken? Are they obstructed?

**Step 2: Adversarial Generation**
   - **Object-Level:** What tool is missing? (e.g., Carpenter scene, no hammer -> "Segment the hammer").
   - **Concept-Level:** Pick a visible object. Find a reason it *cannot* be used. (e.g., A wet bench. Prompt: "Segment the dry surface for sitting").

**Step 3: Verification**
   - Look at the image again. Is the prompt unambiguously false?
   - *Self-Correction:* If asking for "a place to sit", ensure there isn't a small stool in the corner you missed. The function must be totally unavailable.

---

**Output Format (JSON Only):**
[
  {
    "prompt": "Segment the [functional description]",
    "strategy": "object_level_neighbor" OR "concept_level_neighbor",
    "concept_family": "context_dependent",
    "target_decoy": "The object/scene element that makes this tricky (e.g., 'The cluttered chair' or 'The plastic fruit')"
  }
]
"""

PHYSICS_NEGATIVE_META_PROMPT = """
You are an expert Adversarial AI tasked with generating "Hard Negative" segmentation prompts about physics, stability, and safety based on an image.

**TASK:** Design up to 3 deceptive segmentation prompts that describe plausible physical states, hazards, or dynamics, but **do not map to any pixels in the image**. Your goal is to trick a segmentation model that recognizes objects but fails to verify their precise *physical condition* or *interaction with gravity/environment*.

---

### GUIDING PRINCIPLE: The Physical Hallucination Rule

The prompt must trigger the model's "physics priors." It should describe a physical state that is often associated with the scene or object (e.g., "spilled liquid" in a kitchen, "unstable stack" in a warehouse), but is **factually incorrect** in this specific instance because everything is stable, clean, or safe.

**The output mask for all your prompts must be EMPTY.**

---

### STRATEGIES FOR HARD NEGATIVES

You must use a mix of the following two strategies:

**Strategy 1: Object-Level Neighbors (The Missing Hazard/Structure)**
*   **Logic:** Identify the scene context (e.g., Construction Site, messy desk). Generate a prompt for a physical entity (hazard, barrier, support) that is **statistically likely** in this environment but **ABSENT** in the image.
*   **Target:** The model sees "Construction Site" and hallucinates "Falling Debris" or "Safety Barrier".
*   **Example:** Image shows a *Clean floor*. Prompt: "Segment the trip hazard." (Context implies hazards, but floor is safe).
*   **Example:** Image shows a *Table with no coaster*. Prompt: "Segment the protective layer under the cup." (Missing support object).

**Strategy 2: Concept-Level Neighbors (The False Physical State)**
*   **Logic:** Identify an object that **IS PRESENT**. Analyze its actual physical state (Stable, Empty, Dull, Cold, Supported). Generate a prompt that names the object but assigns it a **FALSE** or **OPPOSITE** physical property.
*   **Target:** The model sees a "Tall Vase" and assumes it is "Unstable" without checking the wide base.
*   **Example (Stability):** Image shows a *Wide, heavy box on the floor*. Prompt: "Segment the object likely to tip over." (Object is stable).
*   **Example (Containment):** Image shows an *Empty glass*. Prompt: "Segment the liquid inside the container." (Object is empty).
*   **Example (Hazard):** Image shows a *Round plastic spoon*. Prompt: "Segment the sharp object capable of cutting." (Object is dull).
*   **Example (Temperature):** Image shows an *Ice cube*. Prompt: "Segment the item emitting heat." (Object is cold).

---

### PHYSICS CONCEPT FAMILIES (Use these to craft "False States" or "Missing Hazards")

1.  **support_stability:** Wrong stability (e.g., asking for "precariously balanced" when item is anchored/flat).
2.  **containment_and_flow:** Wrong fill state (e.g., asking for "overflowing" when empty, "leaking" when sealed).
3.  **collision_obstruction_passability:** Wrong blockage (e.g., asking for "the obstacle blocking the path" when the hallway is clear).
4.  **hazard:** Wrong safety assessment (e.g., asking for "slippery surface" on a dry rug, or "electrified wire" on a harmless string).
5.  **counterfactuals:** Wrong prediction (e.g., "object that would roll if released" for a square block).
6.  **unusual:** Wrong anomaly (e.g., asking for "floating object" when everything is obeying gravity).

---

### REASONING PROCESS

**Step 1: Physical Inspection**
   - Analyze the image physics. Is it messy or clean? Stable or chaotic?
   - Identify dominant objects and their *true* physical state (e.g., The ladder is folded and lying flat = Stable).

**Step 2: Adversarial Generation**
   - **Object-Level:** Is there a hazard missing? (e.g., Wet floor sign missing on a dry floor -> "Segment the hazard warning").
   - **Concept-Level:** Pick a visible object. Invert its physics. (e.g., A closed, sealed box. Prompt: "Segment the spilling contents").

**Step 3: Verification**
   - Look at the image again. Is the prompt unambiguously false?
   - *Self-Correction:* If you ask for "something that could break", ensure there isn't a glass in the background. The prompt must target a property that is visually disproven.

---

**Output Format (JSON Only):**
[
  {
    "prompt": "Segment the [object] [physical_state_phrase]",
    "strategy": "object_level_neighbor" OR "concept_level_neighbor",
    "concept_family": "support_stability",
    "target_decoy": "The object/scene element that makes this tricky (e.g., 'The stable box' or 'The dry floor')"
  }
]
"""