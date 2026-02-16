"""Compatibility shim for legacy imports.

Canonical source: `dataengine.prompts.concept_specific_meta_prompts_for_negatives`.
"""

try:
    from dataengine.prompts.concept_specific_meta_prompts_for_negatives import *  # noqa: F401,F403
except ImportError:
    from prompts.concept_specific_meta_prompts_for_negatives import *  # type: ignore # noqa: F401,F403

