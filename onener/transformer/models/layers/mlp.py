import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 hidden_dims: int,
                 hidden_dropout_prob: float,
                 ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.classifier = nn.Linear(hidden_dims, output_dims)

    def forward(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.classifier(x)

        return x
