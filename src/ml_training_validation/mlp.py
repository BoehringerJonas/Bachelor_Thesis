import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, mlf_size:int, tit_size:int, tab_size:int) -> None:
        super().__init__()
        self.mlf_net = nn.Sequential(
            nn.Linear(mlf_size, 64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.tit_net = nn.Sequential(
            nn.Linear(tit_size, 64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

        self.tab_net = nn.Sequential(
            nn.Linear(tab_size, 16),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, mlf:torch.Tensor, tit:torch.Tensor, tab:torch.Tensor) -> torch.Tensor:
        mlf = self.mlf_net(mlf)
        tit = self.tit_net(tit)
        tab = self.tab_net(tab)

        x = torch.cat([mlf, tit, tab], dim=1)
        return self.head(x)