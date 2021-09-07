import logging
from typing import List, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from onener.transformer.pretokenizer import PreTokenizer
from onener.transformer.utils import InputFeaturesCustom


logger = logging.getLogger(__name__)


class OutputExample(TypedDict):
    word: str
    score: float
    entity: str
    index: int


class Pipeline:
    def __init__(self,
                 model,
                 pretokenizer: PreTokenizer,
                 tokenizer,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.pretokenizer = pretokenizer
        self.tokenizer = tokenizer

    def __call__(self, texts: Union[str, List[str]]) -> List[List[OutputExample]]:
        if isinstance(texts, str):
            texts = [texts]

        texts_pretokenized = [self.pretokenizer.parse(text) for text in texts]
        features = self._convert_words_to_feature(
            sentences=texts_pretokenized,
            tokenizer=self.tokenizer
        )

        output = []
        for i, feature in enumerate(features):
            entities = []
            idx_start = 0
            tags, logits = self._predict(feature)

            for idx, (tag, logit) in enumerate(zip(tags, logits)):
                if tag.item() != -1:
                    entity: OutputExample = {
                        "word": texts_pretokenized[i][idx],
                        "score": logit[tag.item()].item(),
                        "entity": self.model.config.id2label[tag.item()],
                        "index": idx_start + 1
                    }
                    entities.extend([entity])
                    idx_start += 1
            output.append(entities)

        return output

    def _predict(self, feature: InputFeaturesCustom) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            logits, tags = self.model(
                input_ids=torch.LongTensor(feature.input_ids).unsqueeze(0).to(self.device),
                attention_mask=torch.LongTensor(feature.attention_mask).unsqueeze(0).to(self.device),
                valid_mask=torch.LongTensor(feature.valid_mask).unsqueeze(0).to(self.device),
                token_type_ids=torch.LongTensor(feature.token_type_ids).unsqueeze(0).to(self.device)
            )

            softmax = nn.Softmax(dim=-1)
            tags, logits = tags.squeeze(0), softmax(logits.squeeze(0))

        return tags, logits

    def _convert_words_to_feature(
        self,
        sentences: List[List[str]],
        tokenizer: PreTrainedTokenizer,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ) -> List[InputFeaturesCustom]:
        features = []

        for words in sentences:
            tokens = []
            valid_mask = []
            for word in words:
                word_tokens = tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)

                for i, word_token in enumerate(word_tokens):
                    if i == 0:
                        valid_mask.append(1)
                    else:
                        valid_mask.append(0)

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            max_seq_length = tokenizer.model_max_length
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                no_predictions = tokenizer.convert_tokens_to_string(
                    tokens[(max_seq_length - special_tokens_count):]
                )
                logger.warning(f"Maximum sequence length exceeded: No prediction for {no_predictions}.")
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            valid_mask.append(1)
            segment_ids = [sequence_a_segment_id] * len(tokens)

            # Put a cls token at the beginning of a sentence
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            valid_mask.insert(0, 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            feature = InputFeaturesCustom(
                input_ids=input_ids,
                valid_mask=valid_mask,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
            )

            features.append(feature)

        return features
