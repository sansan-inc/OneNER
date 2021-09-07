from typing import List, Optional

import torch
import torchcrf


class CRF(torchcrf.CRF):
    def __init__(self,
                 num_tags: int,
                 trans_mask: Optional[List[List[bool]]] = None,  # あり得ない状態遷移をFalseで指定
                 start_trans_mask: Optional[List[bool]] = None,  # あり得ない開始状態をFalseで指定
                 batch_first: bool = False) -> None:
        super().__init__(num_tags=num_tags, batch_first=batch_first)
        self._reset_parameters(trans_mask, start_trans_mask)

    def _reset_parameters(self, trans_mask, start_trans_mask) -> None:
        """
        Add initialization of transition probability to -1e2
        according to the input mask.
        """

        if start_trans_mask is not None:
            self.start_transitions.data = self.start_transitions.masked_fill(
                torch.ByteTensor(start_trans_mask) == False, -1e2
            )
        if trans_mask is not None:
            self.transitions.data = self.transitions.masked_fill(
                torch.ByteTensor(trans_mask) == False, -1e2
            )

    def _viterbi_decode(self,
                        emissions: torch.FloatTensor,
                        mask: torch.ByteTensor
                        ) -> torch.LongTensor:
        best_tags_list = super()._viterbi_decode(emissions=emissions, mask=mask)

        seq_length, _ = mask.shape
        best_tags_list = [item + [-1] * (seq_length - len(item)) for item in best_tags_list]
        
        return torch.LongTensor(best_tags_list)
