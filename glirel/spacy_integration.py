import os
import types
from typing import List, Optional, Union, Dict

import torch
from datasets import Dataset
from spacy.tokens import Doc, Span
from spacy.util import filter_spans, minibatch

from glirel import GLiREL


class SpacyGLiRELWrapper:
    """This wrapper allows GLiREL to be easily used as a spaCy pipeline component.

    Usage:

    .. code-block:: diff

         import spacy

         nlp = spacy.load("en_core_web_sm")
       + nlp.add_pipe("glirel", config={"model": "jackboyla/glirel_beta"})

         text = '''Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the
         Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her
         death in 30 BCE.'''
         doc = nlp(text)

    Example::

        >>> import spacy
        >>> import span_marker
        >>> nlp = spacy.load("en_core_web_sm")
        >>> nlp.add_pipe("glirel", config={"model": "jackboyla/glirel_beta"})
        >>> text = '''Cleopatra VII, also known as Cleopatra the Great, was the last active ruler of the
        ... Ptolemaic Kingdom of Egypt. She was born in 69 BCE and ruled Egypt from 51 BCE until her
        ... death in 30 BCE.'''
        >>> doc = nlp(text)
        >>> doc.ents
        (Cleopatra VII, Cleopatra the Great, 69 BCE, Egypt, 51 BCE, 30 BCE)
        >>> for span in doc.ents:
        ...     print((span, span.label_))
        (Cleopatra VII, 'PERSON')
        (Cleopatra the Great, 'PERSON')
        (69 BCE, 'DATE')
        (Egypt, 'GPE')
        (51 BCE, 'DATE')
        (30 BCE, 'DATE')
        >>> doc._.relations

    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        labels: List[str],
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> None:
        """Initialize a SpanMarker wrapper for spaCy.

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): The path to a locally pretrained SpanMarker model
                or a model name from the Hugging Face hub, e.g. `tomaarsen/span-marker-roberta-large-ontonotes5`
            batch_size (int): The number of samples to include per batch. Higher is faster, but requires more memory.
                Defaults to 4.
            device (Optional[Union[str, torch.device]]): The device to place the model on. Defaults to None.
            overwrite_entities (bool): Whether to overwrite the existing entities in the `doc.ents` attribute.
                Defaults to False.
        """
        self.model = GLiREL.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        self.labels = labels
        if device:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to("cuda")
        self.batch_size = batch_size


    def _set_relatons(self, doc: Doc, relations: List[Dict]):
        doc.set_extension("relations", default=None, force=True)
        doc._.relations = relations
        return doc

    def __call__(self, doc: Doc, labels: List[str] = None) -> Doc:
        # TODO: should chunk tokens if text too long?
        if len(doc.ents) > 1: 
            print("The input text must contain at least two entities; skipping...")
            return doc
        
        if labels is None:
            labels = self.labels
        
        tokens = [token.text for token in doc]
        ner = [[ent.start, (ent.end - 1), ent.label_, ent.text] for ent in doc.ents]
        relations = self.model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=1)


        doc = self._set_relatons(doc, relations)

        return doc

    # def pipe(self, stream, batch_size=128):
    #     ...