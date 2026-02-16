"""Compatibility shim for legacy imports.

Canonical source: `dataengine.prompts.region_dense_caption_meta_prompt`.
"""

try:
    from dataengine.prompts.region_dense_caption_meta_prompt import *  # noqa: F401,F403
except ImportError:
    from prompts.region_dense_caption_meta_prompt import *  # type: ignore # noqa: F401,F403

