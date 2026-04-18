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

        Tests
        ----------
        Runs with ReLU(), GeLU, SiLU. tested different combinations. 
        In the end ist shows that the mix of GeLU and SiLU works best. 
        When changing GeLU to ReLU it doesn`t makes a big diferance, 
        but the GeLu is slightly better.

        Runs with a fusion layer was worse.

        Does one run with gate- and fusion-Layer together and it was way worse.
        """

        super().__init__()
        self.mlf_net = nn.Sequential(
            nn.Linear(mlf_size, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        self.tit_net = nn.Sequential(
            nn.Linear(tit_size, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        self.tab_net = nn.Sequential(
            nn.Linear(tab_size, 16),
            nn.GELU(),#nn.ReLU(),
            nn.LayerNorm(16)
        )

        self.gate = nn.Sequential(
            nn.Linear(64 + 64 + 16, 3),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential( 
            nn.Linear(64 + 64 + 16, 128),
            nn.GELU(),#nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU()#nn.ReLU()
        )
        
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 16, 32),
            nn.GELU(),#nn.ReLU(),
            nn.Linear(32, 1)
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

        a = torch.cat([mlf, tit, tab], dim=1)

        #g = self.gate(a)  # (B, 3)

        #g_mlf = g[:, 0:1]
        #g_tit = g[:, 1:2]
        #g_tab = g[:, 2:3]

        #fused = torch.cat([
        #    g_mlf * mlf,
        #    g_tit * tit,
        #    g_tab * tab
        #], dim=1)

        #x = self.fusion(a)
        return self.head(a)
