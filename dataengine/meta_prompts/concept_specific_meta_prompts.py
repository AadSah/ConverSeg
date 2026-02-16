"""Compatibility shim for legacy imports.

Canonical source: `dataengine.prompts.concept_specific_meta_prompts`.
"""

try:
    from dataengine.prompts.concept_specific_meta_prompts import *  # noqa: F401,F403
except ImportError:
    from prompts.concept_specific_meta_prompts import *  # type: ignore # noqa: F401,F403

