import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, mlf_size:int, tit_size:int, tab_size:int) -> None:
        """
        Initiates the MLP Layers. 3 Input layers for MLF, TIT and Tab and a head to get the output.

        Parameters
        ----------
        mlf_size    : int
            Size of MLF features
        tit_size    : int
            Size of TIT features
        tab_size    : int
            Size of Tab features
        """
        super().__init__()
        self.mlf_net = nn.Sequential(
            nn.Linear(mlf_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.tit_net = nn.Sequential(
            nn.Linear(tit_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
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
        """
        Training

        Parameters
        ----------
        mlf : torch.Tensor
            MLF features
        tit : torch.Tensor
            TIT features
        tab : torch.Tensor
            Tab data

        Returns
        -------
        torch.tensor : trained model

        """
        mlf = self.mlf_net(mlf)
        tit = self.tit_net(tit)
        tab = self.tab_net(tab)

        x = torch.cat([mlf, tit, tab], dim=1)
        return self.head(x)