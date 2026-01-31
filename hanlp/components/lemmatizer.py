# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-08 18:35
from typing import List, Dict, Any

from hanlp.common.transform import TransformList
from hanlp.components.mtl.tasks.pos import TransformerTagging
from hanlp.components.parsers.ud.lemma_edit import gen_lemma_rule, apply_lemma_rule
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp_common.document import Document


def add_lemma_rules_to_sample(sample: dict):
    if 'tag' in sample and 'lemma' not in sample:
        lemma_rules = [gen_lemma_rule(word, lemma)
                       if lemma != "_" else "_"
                       for word, lemma in zip(sample['token'], sample['tag'])]
        sample['lemma'] = sample['tag'] = lemma_rules
    return sample


class TransformerLemmatizer(TransformerTagger):

    def __init__(self, **kwargs) -> None:
        """A transition based lemmatizer using transformer as encoder.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)

    def build_dataset(self, data, transform=None, **kwargs):
        if not isinstance(transform, list):
            transform = TransformList()
        transform.append(add_lemma_rules_to_sample)
        return super().build_dataset(data, transform, **kwargs)

    def prediction_to_human(self, pred, vocab: List[str], batch, token=None):
        if token is None:
            token = batch['token']
        rules = super().prediction_to_human(pred, vocab, batch)
        for token_per_sent, rule_per_sent in zip(token, rules):
            lemma_per_sent = [apply_lemma_rule(t, r) for t, r in zip(token_per_sent, rule_per_sent)]
            for i, (t, l) in enumerate(zip(token_per_sent, lemma_per_sent)):
                if t.isdigit():
                    lemma_per_sent[i] = t
            yield lemma_per_sent


class TransformerTaggingLemmatizer(TransformerTagging):

    def transform_batch(self, batch: Dict[str, Any], results: Dict[str, Any] = None, cls_is_bos=False,
                        sep_is_eos=False) -> Dict[str, Any]:
        return batch


    def finalize_document(self, doc: Document, task_name: str):
        tok = doc.get_by_prefix('tok')
        if tok:
            for tokens, lemmas in zip(tok, doc.get_by_prefix('lem')):
                if len(''.join(tokens)) == len(''.join(lemmas)):
                    # Map lemmas into tokens
                    mapped_lemmas = []
                    offset = 0
                    for token in tokens:
                        mapped_lemmas.append(''.join(lemmas[offset:offset + len(token)]))
                        offset += len(token)
                    lemmas.clear()
                    lemmas.extend(mapped_lemmas)
        super().finalize_document(doc, task_name)
