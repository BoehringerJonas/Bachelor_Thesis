import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("feature")


class FeatureZone:
    """
    Pipeline for extracting spatial and temporal features from patient EIT data.

    This class implements a feature extraction pipeline that converts raw
    electrical impedance tomography (EIT) sequences into compact feature
    representations suitable for downstream machine learning models.

    Pipeline Overview
    -----------------
    1. **Data Loading**
       - Loads cleaned patient EIT sequences (MLM and TIT datasets) from disk.
       - Each patient has separate files for EIT images and optionally
         pressure and volume signals.

    2. **Spatial Feature Extraction (ResNet-50)**
       - Grayscale EIT images are converted to 3-channel tensors:
           * 3D variant: replicate the grayscale image across all three channels.
           * VP variant: channel 1 = EIT, channel 2 = pressure, channel 3 = volume.
       - Images are resized to 224×224 and passed through a **frozen ResNet-50** 
         backbone to obtain per-frame embeddings of shape (2048,).

    3. **Temporal Feature Aggregation (LSTM)**
       - Per-frame spatial embeddings are fed into a 2-layer **LSTM** (hidden size 512).
       - The final hidden state of the last layer represents the temporal feature
         sequence for the patient.
       - Output is a fixed-length vector of shape (512,).

    4. **Saving Features**
       - Features are saved in compressed `.npz` files for each patient:
           * `Res_*id*.npz`: raw ResNet features
           * `LSTM_*id*.npz`: temporal LSTM features
       - Separate files are created for MLM (maximal flow maneuver) and
         TIT (titration) datasets, as well as for 3D and VP variants.

    5. **Design Notes**
       - Networks are **frozen**, ensuring deterministic feature extraction.
       - Batching is applied to reduce memory usage during feature extraction.
       - The pipeline does **not modify target variables**; it focuses exclusively
         on producing embeddings from raw input sequences.
       - This design supports later multi-modal or intermediate fusion architectures
         by providing aligned spatial-temporal features for each patient.

    Attributes
    ----------
    path : Path
        Base directory containing cleaned EIT data.
    device : torch.device
        CUDA or CPU device used for computations.
    resnet : nn.Module
        Frozen ResNet-50 backbone for spatial feature extraction.
    lstm : nn.LSTM
        Two-layer LSTM for temporal aggregation of spatial features.
    """

    def __init__(self, path:Path) -> None:
        """
        Selects a CUDA-capable NVIDIA GPU if available; otherwise, uses the CPU.
        Initializes the ResNet and LSTM networks for feature extraction.

        Input:
        path : Path
            Path to the input data.
        """
        self.path = Path(path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info("def ResNet")
        self.resnet = self.get_res().to(self.device)

        logger.info("def LSTM")
        self.lstm = torch.nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        ).to(self.device)


    def run(self) -> None:
        """
        Iterates over the cleaned patients and extracts features using ResNet followed by
        LSTM, producing image-based features (ResNet) and temporal features (LSTM).
        The MLM (maximum flow maneuver) and TIT (titration) are used as separate inputs.

        To enable comparative feature evaluation, two variants are generated:
        a 3D version, where grayscale images are replicated across all three RGB channels,
        and a VD version, where the second channel represents pressure and the third channel
        represents volume from the time series to construct RGB images.

        Each MLM and titration feature set is saved to a file in a new feature directory.
        Files named `Res_<id>.npz` contain features after ResNet extraction, and
        `LSTM_<id>.npz` contain the final features after LSTM processing.
        """

        torch.set_grad_enabled(False)
        self.resnet.eval()
        self.lstm.eval()

        csv_path = self.path/ 'raw' / 'MLM' / 'peep_tabular_data.csv'
        table = pd.read_csv(csv_path)
        table = table[~table["Pat-ID"].str.startswith("AD")]
        table = table.set_index("Pat-ID")
        for id in table.index:
            logger.info("Starting itteration for: " + str(table.loc[id, "PEEP"]))
            logger.info("3D")
            logger.info("MLM")
            mlmPath = self.path / "clean" / id / "MLMeit.npz"

            if not mlmPath.exists():
                logger.warning(f"Missing file, skipping: {mlmPath}")
                continue

            mlmData = np.load(mlmPath)
            mlmRes3D = self.extract_resnet_features3D(mlmData['EITdata'])
            mlmLSTM3D = self.extract_LSTM_features(mlmRes3D)

            logger.info("TIT")
            titPath = self.path / "clean" / id / "TITeit.npz"
            titData = np.load(titPath)
            titRes3D = self.extract_resnet_features3D(titData['EITdata'])
            titLSTM3D = self.extract_LSTM_features(titRes3D)

            logger.info("save")
            save_path = Path(self.path) / 'feature' / "3D"
            save_path.mkdir(parents = True, exist_ok = True)

            np.savez_compressed(
                save_path / f"3Res_{id}.npz",
                patient_id = id,
                mlm = mlmRes3D.cpu().numpy(),
                tit = titRes3D.cpu().numpy(),
            )

            np.savez_compressed(
                save_path / f"3DLSTM_{id}.npz",
                patient_id = id,
                mlm = mlmLSTM3D.cpu().numpy(),
                tit = titLSTM3D.cpu().numpy(),
            )
            del mlmLSTM3D, titLSTM3D

            logger.info("vp")
            mlmResVP = self.extract_resnet_featuresVP(mlmData['EITdata'], mlmData['pressureData'], mlmData['volumeData'])
            mlmLSTMVP = self.extract_LSTM_features(mlmResVP)

            titResVP = self.extract_resnet_featuresVP(titData['EITdata'], titData['pressureData'], titData['volumeData'])
            titLSTMVP = self.extract_LSTM_features(titResVP)

            logger.info("save")
            save_path = Path(self.path) / 'feature' / "VD"
            save_path.mkdir(parents = True, exist_ok = True)

            np.savez_compressed(
                save_path / f"VDRes_{id}.npz",
                patient_id = id,
                mlm = mlmResVP.cpu().numpy(),
                tit = titResVP.cpu().numpy(),
            )

            np.savez_compressed(
                save_path / f"VDLSTm_{id}.npz",
                patient_id = id,
                mlm = mlmLSTMVP.cpu().numpy(),
                tit = titLSTMVP.cpu().numpy(),
            )


    def get_res(self) -> nn.Module:
        """
        Loads a ResNet-50 model pretrained on ImageNet, removes the final classification layer,
        and freezes all parameters. The returned module can be used as a fixed feature extractor
        for downstream models.

        Returns:
        nn.Module
            A ResNet-50 backbone without the final fully connected layer, with all parameters
            frozen, providing only the output feature vector.
        """

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        for p in resnet.parameters():
            p.requires_grad = False
        return resnet


    def extract_resnet_features3D(self, frames:np.ndarray, batch_size:int=256) -> torch.Tensor:
        """
        Extracts spatial features from a temporal sequence of grayscale images
        using a frozen ResNet-50 backbone.

        The input sequence is processed in batches to manage memory usage.
        Each grayscale frame is replicated across three channels to form an RGB tensor,
        resized to (224, 224) using bilinear interpolation, and passed through the
        pretrained ResNet-50 feature extractor. The resulting per-frame feature vectors
        are concatenated to form a temporal feature sequence.

        Input:
        frames : np.ndarray
            Input image sequence of shape (T, H, W), where T is the number of time steps.
        batch_size : int, optional
            Number of frames to process per batch.

        Returns:
        torch.Tensor
            Array of shape (T, 2048) containing the extracted ResNet features, with T
            representing the temporal dimension.
        """

        logger.info("resnet features")
        assert frames.ndim == 3, f"Expected (T,H,W), got {frames.shape}"

        spatial_features = []

        with torch.no_grad():
            for i in range(0, frames.shape[0], batch_size):
                chunk = torch.from_numpy(frames[i:i+batch_size]) \
                            .float() \
                            .to(self.device, non_blocking=True)

                chunk = chunk.unsqueeze(1)           # (B,1,H,W)
                chunk = chunk.expand(-1, 3, -1, -1)  # (B,3,H,W)

                chunk = F.interpolate(
                    chunk,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                features = self.resnet(chunk)         # (B,2048,1,1)
                features = features.flatten(1)        # (B,2048)

                spatial_features.append(features)

        return torch.cat(spatial_features, dim=0)
    

    def extract_LSTM_features(self, spatial_seq:torch.Tensor) -> torch.Tensor:
        """
        Extracts a fixed-length temporal feature representation from a sequence
        of spatial features using an LSTM.

        The input sequence of per-frame feature vectors is treated as a temporal sequence
        and passed through an LSTM. The final hidden state of the last LSTM layer is
        returned as a compact representation of the entire sequence.

        Input:
        spatial_seq : torch.Tensor
            Input sequence of shape (T, 2048), where T is the number of time steps.

        Returns:
        torch.Tensor
            Tensor of shape (H,), where H is the LSTM hidden size (e.g., 512), representing
            the encoded temporal feature.
        """
        logger.info("LSTM features")
        spatial_seq = spatial_seq.to(self.device)
        seq = spatial_seq.unsqueeze(0)              # (1, T, 2048)
        out, (h_n, c_n) = self.lstm(seq)

        return h_n[-1].squeeze(0)                   # (512,)


    def extract_resnet_featuresVP(self, frames:np.ndarray, pre:np.ndarray, vol:np.ndarray, batch_size=256) -> torch.Tensor:
        """
        Extracts spatial features from a temporal sequence of grayscale images
        using a frozen ResNet-50 backbone.

        The input sequence is processed in batches to manage memory usage. An RGB tensor
        is constructed for each frame, where the first channel contains the grayscale image,
        the second channel contains the pressure, and the third channel contains the volume
        of the given sequence. Each tensor is resized to (224, 224) using bilinear interpolation
        and passed through a pretrained ResNet-50 feature extractor. The resulting per-frame
        feature vectors are concatenated to form a temporal feature sequence.

        Input:
        frames : np.ndarray
            Input image sequence of shape (T, H, W), where T is the number of time steps.
        batch_size : int, optional
            Number of frames processed per batch.

        Returns:
        torch.Tensor
            Tensor of shape (T, 2048) containing the extracted ResNet features, with T
            representing the temporal dimension.
        """
        logger.info("resnet VP features")
        assert frames.ndim == 3, f"Expected (T,H,W), got {frames.shape}"
        assert pre.ndim == 2, f"Expected (1, P), got {pre.shape}"
        assert vol.ndim == 2, f"Expected (1, V), got {vol.shape}"
        assert frames.shape[0] == pre.shape[1] == vol.shape[1], "Frames, Volume, Pressure do not have the same length"

        spatial_features = []

        with torch.no_grad():
            for i in range(0, frames.shape[0], batch_size):
                chunk = torch.from_numpy(frames[i:i+batch_size]) \
                    .float() \
                    .to(self.device, non_blocking=True) \
                    .unsqueeze(1)                   # (B,1,H,W)

                pChunk = torch.from_numpy(pre[0, i:i+batch_size].astype(np.float32)) \
                    .float() \
                    .to(self.device, non_blocking=True) \
                    .view(-1,1,1,1) \
                    .expand(-1,1,chunk.shape[2],chunk.shape[3])

                vChunk = torch.from_numpy(vol[0, i:i+batch_size].astype(np.float32)) \
                    .float() \
                    .to(self.device, non_blocking=True) \
                    .view(-1, 1, 1, 1) \
                    .expand(-1, 1, chunk.shape[2], chunk.shape[3])

                vp = torch.cat([chunk, pChunk, vChunk], dim=1)
                vp = F.interpolate(
                    vp,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                features = self.resnet(vp)          # (B,2048,1,1)
                features = features.flatten(1)      # (B,2048)

                spatial_features.append(features)

        return torch.cat(spatial_features, dim=0)
