import torch
from typing import Tuple, Optional


def valid_sequence_output(sequence_output: torch.Tensor,
                          valid_mask: torch.Tensor,
                          attention_mask: torch.Tensor,
                          labels: Optional[torch.Tensor] = None,
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    "給", "食"のような"食"の部分を無視して詰めてしまう関数
    もともと、"給食"のような1単語のもの(教師ラベルも1つ)をAutoTokenizerで、"給"と"食"にしたため、
    "給食"の代表として、"給"のベクトルを使って、学習するため
    """

    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                               device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long,
                                       device='cuda' if torch.cuda.is_available() else 'cpu')

    if labels is not None:
        valid_labels = torch.zeros(batch_size, max_len, dtype=torch.long,
                                   device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        valid_labels = None

    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
                if labels is not None:
                    valid_labels[i][jj] = labels[i][j]
    return valid_output, valid_labels, valid_attention_mask


def remove_special_output(sequence_output: torch.Tensor,
                          attention_mask: torch.Tensor,
                          labels: torch.Tensor = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [CLS], [SEP]の部分を無視して詰めてしまう関数
    """

    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                               device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long,
                                       device='cuda' if torch.cuda.is_available() else 'cpu')

    if labels is not None:
        valid_labels = torch.zeros(batch_size, max_len, dtype=torch.long,
                                   device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        valid_labels = None

    for i in range(batch_size):
        jj = -1
        real_seq_len = torch.sum(attention_mask[i], dim=-1) - 2
        for j in range(1, real_seq_len+1):
            jj += 1
            valid_output[i][jj] = sequence_output[i][j]
            valid_attention_mask[i][jj] = attention_mask[i][j]
            if labels is not None:
                valid_labels[i][jj] = labels[i][j]

    return valid_output, valid_labels, valid_attention_mask
