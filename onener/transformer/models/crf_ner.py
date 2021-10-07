import torch.nn as nn

from transformers.modeling_bert import BertModel
from transformers.modeling_bert import BertPreTrainedModel

from .layers.crf import CRF
from .utils import valid_sequence_output, remove_special_output


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if kwargs.get("transition_parameter_init"):
            labels = config.label2id.keys()
            trans_mask = []
            for before_tag in labels:
                trans_mask_ = []
                for after_tag in labels:
                    if "I" in after_tag:
                        if after_tag.split("-")[1] in before_tag:
                            trans_mask_.append(True)
                        else:
                            trans_mask_.append(False)
                    else:
                        trans_mask_.append(True)

                trans_mask.append(trans_mask_)

            start_trans_mask = [False if "I" in k else True for k in config.label2id.keys()]
        else:
            start_trans_mask = None
            trans_mask = None

        self.crf = CRF(num_tags=config.num_labels,
                       batch_first=True,
                       trans_mask=trans_mask,
                       start_trans_mask=start_trans_mask)
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        sequence_output, labels, attention_mask = valid_sequence_output(
            sequence_output=sequence_output,
            valid_mask=valid_mask,
            attention_mask=attention_mask,
            labels=labels
        )
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        logits, labels, attention_mask = remove_special_output(
            sequence_output=logits,
            labels=labels,
            attention_mask=attention_mask
        )

        if self.training:
            outputs = (logits,)
        else:
            tags = self.crf.decode(logits, attention_mask.bool())
            outputs = (tags,)

        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.bool())
            outputs = (-1 * loss,) + outputs
        else:
            outputs = (logits,) + outputs

        return outputs  # (loss), scores
