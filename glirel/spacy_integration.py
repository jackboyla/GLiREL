import os
import types
from typing import List, Optional, Union, Dict

import torch
from datasets import Dataset
from spacy.tokens import Doc, Span
from spacy.util import filter_spans, minibatch
from loguru import logger

from glirel import GLiREL
from glirel.modules.utils import constrain_relations_by_entity_type


class SpacyGLiRELWrapper:
    """This wrapper allows GLiREL to be easily used as a spaCy pipeline component.

    Usage:

        # Add the GLiREL component to the pipeline
        >>> nlp.add_pipe("glirel", after="ner")

        # Now you can use the pipeline with the GLiREL component
        >>> text = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California."

        >>> labels = {"glirel_labels": {
            'co-founder': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]}, 
            'country of origin': {"allowed_head": ["PERSON", "ORG"], "allowed_tail": ["LOC", "GPE"]}, 
            'licensed to broadcast to': {"allowed_head": ["ORG"]},  
            'no relation': {},  
            'parent': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]}, 
            'followed by': {"allowed_head": ["PERSON", "ORG"], "allowed_tail": ["PERSON", "ORG"]},  
            'located in or next to body of water': {"allowed_head": ["LOC", "GPE", "FAC"], "allowed_tail": ["LOC", "GPE"]},  
            'spouse': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
            'child': {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},  
            'founder': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},  
            'founded on date': {"allowed_head": ["ORG"], "allowed_tail": ["DATE"]},
            'headquartered in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},  
            'acquired by': {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},  
            'subsidiary of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]}, 
            }
        }

        >>> docs = list( nlp.pipe([(text, labels)], as_tuples=True) )
        >>> relations = docs[0][0]._.relations

        >>> sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
        >>> print("\nDescending Order by Score:")
        >>> for item in sorted_data_desc:
            print(item)

        >>> Descending Order by Score:
        {'head_pos': [0, 2], 'tail_pos': [25, 26], 'head_text': ['Apple', 'Inc.'], 'tail_text': ['California'], 'label': 'headquartered in', 'score': 0.9854260683059692}
        {'head_pos': [0, 2], 'tail_pos': [23, 24], 'head_text': ['Apple', 'Inc.'], 'tail_text': ['Cupertino'], 'label': 'headquartered in', 'score': 0.9569844603538513}
        {'head_pos': [8, 10], 'tail_pos': [0, 2], 'head_text': ['Steve', 'Wozniak'], 'tail_text': ['Apple', 'Inc.'], 'label': 'co-founder', 'score': 0.09025496244430542}
        {'head_pos': [5, 7], 'tail_pos': [0, 2], 'head_text': ['Steve', 'Jobs'], 'tail_text': ['Apple', 'Inc.'], 'label': 'co-founder', 'score': 0.08805803954601288}
        {'head_pos': [12, 14], 'tail_pos': [0, 2], 'head_text': ['Ronald', 'Wayne'], 'tail_text': ['Apple', 'Inc.'], 'label': 'co-founder', 'score': 0.07996643334627151}

    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.3,
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
        if device:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to("cuda")
        self.batch_size = batch_size
        self.threshold = threshold

    def _set_relatons(self, doc: Doc, relations: List[Dict]):
        doc.set_extension("relations", default=None, force=True)
        doc._.relations = relations
        return doc

    def __call__(self, doc: Doc, threshold=None) -> Doc:
        threshold = threshold or self.threshold
        if len(doc.ents) < 2: 
            logger.warning("The input text must contain at least two entities; skipping...")
            doc = self._set_relatons(doc, relations=[])
            return doc

        try:
            labels = doc._context["glirel_labels"]
        except Exception as e:
            logger.error("The labels must be passed as a context attribute eg, `nlp.pipe([(text, {'re_labels': ['father', ..]})], as_tuples=True)`")
            raise e

        if isinstance(labels, dict):
            labels_and_constraints = labels
            labels = list(labels.keys())
        
        tokens = [token.text for token in doc]
        ner = [[ent.start, (ent.end - 1), ent.label_, ent.text] for ent in doc.ents]
        relations = self.model.predict_relations(tokens, labels, threshold=threshold, ner=ner, top_k=1)

        if isinstance(doc._context["glirel_labels"], list) is False:
            relations = constrain_relations_by_entity_type(doc.ents, labels_and_constraints, relations)

        doc = self._set_relatons(doc, relations)
        return doc

    # def pipe(self, stream, batch_size=128):
    #     ...