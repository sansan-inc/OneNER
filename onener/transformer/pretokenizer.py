from abc import ABC, abstractmethod
from typing import List

import MeCab


class PreTokenizer(ABC):
    """AutoTokenizer(bert-base-multilingual-cased)によって分かち書きをする前に、
    前処理としてかける分かち書きのための抽象クラス
    これを行わないと場合によっては、
    「〇〇は3月2日に開催した」 -> 「〇 〇 は3 月 2 日 に 開 催 した」のように、「は3」のような
    単語になってしまい、その後の分類タスクで悪影響が起きてしまう
    """
    @abstractmethod
    def parse(self, text: str) -> List[str]:
        raise NotImplementedError


class MeCabPreTokenizer(PreTokenizer):
    """AutoTokenizerによって、分かち書きをする前にMeCabを使用して分かち書きをする場合に使用するクラス"""
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")

    def parse(self, text: str) -> List[str]:
        return self.tagger.parse(text).split()
