import torch.nn as nn
from .constant import CONTROLLER_INPUT_SIZE, CONTROLLER_HIDDEN_SIZE
class Controller(nn.Module):
    def __init__(self, pmf_sizes: tuple[int]):
        super().__init__()

        self.splits = [0]
        for size in pmf_sizes:
            self.splits.append(self.splits[-1]+size)
        self.net = nn.Sequential(
            nn.Linear(CONTROLLER_INPUT_SIZE, CONTROLLER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(CONTROLLER_HIDDEN_SIZE, self.splits[-1])
        )
    
    def forward(self, x):
        logits = self.net(x)

        pmfs = []
        for i in range(len(self.splits)-1):
            pmf = F.softmax(logits[self.splits[i]:self.splits[i+1]], dim=0)
            pmfs.append(pmf)
        return pmfs

