import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class multi_Linear(nn.Module):
    """
    2-layer MLP with ReLU activation.
    """

    def __init__(
        self, input_size: int = 512, hidden_size: int = 512, output_size: int = 1000
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
class MLP(nn.Module):
    """
    2-layer MLP with ReLU activation.
    """

    def __init__(
        self, input_size: int = 512, hidden_size: int = 512, output_size: int = 1000
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Linear(nn.Module):
    """
    Linear layer.
    """

    def __init__(
        self, input_size: int = 512, output_size: int = 1000, bias: bool = True
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


class outlier_MLP(nn.Module):

    def __init__(
            self
    ) -> None:
        super().__init__()
        self.outlier_MLP = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.outlier_MLP(x)
        return x


