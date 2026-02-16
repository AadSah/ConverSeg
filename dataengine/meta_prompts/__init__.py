"""
Backward-compatible import surface for legacy `dataengine.meta_prompts`.

Canonical prompt sources now live in `dataengine.prompts`.
"""

from . import concept_specific_meta_prompts
from . import concept_specific_meta_prompts_for_negatives
from . import region_dense_caption_meta_prompt
