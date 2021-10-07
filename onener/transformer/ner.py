import random
import re
from typing import Any, List, Generator

import torch
import numpy as np
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

from onener.transformer.auto_model import AutoModelForCrfNer
from onener.transformer.pipeline import OutputExample, Pipeline
from onener.transformer.pretokenizer import MeCabPreTokenizer


Tokens = List[dict]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(1)


class BertNERTagger:
    def __init__(self, model_path: str):
        self.pipeline = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(model_path),
            tokenizer=AutoTokenizer.from_pretrained(
                model_path,
                do_lower_case=False
            ),
            ignore_labels=[]
        )
        self.special_tokens = set(self.pipeline.tokenizer.special_tokens_map.values())

    def predict(self, sentences: List[str]) -> List[dict]:
        return list(self._predict(sentences))

    def _predict(self, sentences: List[str]) -> Generator:
        answers = list(self.pipeline(sentences))
        for tokens in _align_dimensions(answers):
            tokens = self._remove_special_token(tokens)
            yield self._remove_wordpiece_special_character(tokens)

    def _remove_special_token(self, tokens: List[dict]) -> Tokens:
        return [t for t in tokens if t["word"] not in self.special_tokens]

    def _remove_wordpiece_special_character(self, tokens: List[dict]) -> Tokens:
        removed_tokens = []
        for token in tokens:
            token["word"] = remove_wordpiece_special_character(token["word"])
            removed_tokens.append(token)
        return removed_tokens


def _align_dimensions(answers: List[Any]) -> List[Tokens]:
    """pipelineは入力が一文の場合は、出力の次元数を一つ落とす処理が書かれている。
    処理を統一するために次元数を統一する。

    Args:
        answers (List): transformers の出力

    Returns:
        List[Tokens]: 統一された次元数の解析結果
    """
    if isinstance(answers[0], dict):
        answers = [answers]
    return answers


def remove_wordpiece_special_character(word: str) -> str:
    """WordPiece の特殊なトークンを削除する

    Args:
        word (str): WordPiece で分割された単語

    Returns:
        str: WordPiece の特殊なトークンが削除された単語
    """
    return re.sub(r"^##", "", word)


class BertCrfNERTagger:
    def __init__(self, model_path: str):
        self.pipeline = Pipeline(
            model=AutoModelForCrfNer.from_pretrained(model_path),
            pretokenizer=MeCabPreTokenizer(),
            tokenizer=AutoTokenizer.from_pretrained(
                model_path,
                do_lower_case=False
            )
        )

    def predict(self, sentences: str) -> List[OutputExample]:
        return self.pipeline(sentences)
