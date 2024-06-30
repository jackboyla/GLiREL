__version__ = "0.1.3"

from .model import GLiREL
from typing import Optional, Union, List
import torch

__all__ = ["GLiREL"]


# https://github.com/tomaarsen/SpanMarkerNER/blob/main/span_marker/__init__.py
# Set up for spaCy
try:
    from spacy.language import Language
except ImportError:
    pass
else:

    DEFAULT_SPACY_CONFIG = {
        "model": "jackboyla/glirel_beta",
        "batch_size": 1,
        "device": None,
    }

    @Language.factory(
        "glirel",
        assigns=["doc._.relations"],
        default_config=DEFAULT_SPACY_CONFIG,
    )
    def _spacy_glirel_factory(
        nlp: Language,
        name: str, 
        model: str,
        batch_size: int,
        device: Optional[Union[str, torch.device]],
    ) -> "SpacyGLiRELWrapper":
        from glirel.spacy_integration import SpacyGLiRELWrapper
        return SpacyGLiRELWrapper(model, batch_size=batch_size, device=device)