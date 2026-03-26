import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple



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
       - Loads cleaned patient EIT sequences (MLF and TIT datasets) from disk.
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
       - Separate files are created for MLF (maximal flow maneuver) and
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

    def __init__(self, path:Path, split:str, set:str) -> None:
        """
        Selects a CUDA-capable NVIDIA GPU if available; otherwise, uses the CPU.
        Initializes the ResNet and LSTM networks for feature extraction.

        Input:
        path : Path
            Path to the input data.
        """
        self.path = path
        self.split = split
        self.set = set

        if split == "train":
            self.MLFmin, self.MLFmax, self.MLFpmean, self.MLFpstd, self.MLFvmean, self.MLFvstd = self.get_global("MLF")
            self.TITmin, self.TITmax, self.TITpmean, self.TITpstd, self.TITvmean, self.TITvstd = self.get_global("TIT")

            self.amean, self.astd, self.bmean, self.bstd, self.wmean, self.wstd, self.hmean, self.hstd, self.pmean, self.pstd = self.get_tglobal()

            spath = self.path / "feature" / self.set / "stats"
            spath.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                spath / "train_stats.npz",

                MLFmin = self.MLFmin,
                MLFmax = self.MLFmax,
                MLFpmean = self.MLFpmean,
                MLFpstd = self.MLFpstd,
                MLFvmean = self.MLFvmean,
                MLFvstd = self.MLFvstd,

                TITmin = self.TITmin,
                TITmax = self.TITmax,
                TITpmean =self.TITpmean,
                TITpstd = self.TITpstd,
                TITvmean = self.TITvmean,
                TITvstd = self.TITvstd,

                amean = self.amean,
                astd = self.astd,
                bmean = self.bmean,
                bstd = self.bstd,
                wmean = self.wmean,
                wstd = self.wstd,
                hmean = self.hmean,
                hstd = self.hstd,
                pmean = self.pmean,
                pstd = self.pstd,
            )
        else:
            spath = self.path / "feature" / self.set / "stats"
            assert spath.exists(), (
                "train_stats.npz not found. "
                "You must run FeatureZone with split='train' first."
            )

            stats = np.load(spath / "train_stats.npz")

            self.MLFmin = stats["MLFmin"]
            self.MLFmax  = stats["MLFmax"]
            self.MLFpmean = stats["MLFpmean"]
            self.MLFpstd  = stats["MLFpstd"]
            self.MLFvmean = stats["MLFvmean"]
            self.MLFvstd  = stats["MLFvstd"]

            self.TITmin = stats["TITmin"]
            self.TITmax  = stats["TITmax"]
            self.TITpmean = stats["TITpmean"]
            self.TITpstd  = stats["TITpstd"]
            self.TITvmean = stats["TITvmean"]
            self.TITvstd  = stats["TITvstd"]

            self.amean = stats["amean"]
            self.astd  = stats["astd"]
            self.bmean = stats["bmean"]
            self.bstd  = stats["bstd"]
            self.wmean = stats["wmean"]
            self.wstd  = stats["wstd"]
            self.hmean = stats["hmean"]
            self.hstd  = stats["hstd"]
            self.pmean = stats["pmean"]
            self.pstd  = stats["pstd"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")
        self.MLFmin = torch.tensor(self.MLFmin, device=self.device)
        self.MLFmax = torch.tensor(self.MLFmax, device=self.device)

        self.TITmin = torch.tensor(self.TITmin, device=self.device)
        self.TITmax = torch.tensor(self.TITmax, device=self.device)

        self.MLFpmean = torch.tensor(self.MLFpmean, device=self.device)
        self.MLFpstd  = torch.tensor(self.MLFpstd, device=self.device)
        self.MLFvmean = torch.tensor(self.MLFvmean, device=self.device)
        self.MLFvstd  = torch.tensor(self.MLFvstd, device=self.device)

        self.TITpmean = torch.tensor(self.TITpmean, device=self.device)
        self.TITpstd  = torch.tensor(self.TITpstd, device=self.device)
        self.TITvmean = torch.tensor(self.TITvmean, device=self.device)
        self.TITvstd  = torch.tensor(self.TITvstd, device=self.device)

        logger.info("def ResNet")
        self.resnet = self.get_res().to(self.device) #Init encoder from ResNet

        logger.info("def LSTM")
        self.lstm = torch.nn.LSTM(  #Init encoder from LSTM
            input_size=256,
            hidden_size=128, #testen von 64
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(self.device)

        self.resnet_reduce = nn.Linear(2048, 256).to(self.device)


    def run(self) -> None:
        """
        Iterates over the cleaned patients and extracts features using ResNet followed by
        LSTM, producing image-based features (ResNet) and temporal features (LSTM).
        The MLF (maximum flow maneuver) and TIT (titration) are used as separate inputs.

        To enable comparative feature evaluation, two variants are generated:
        a 3D version, where grayscale images are replicated across all three RGB channels,
        and a VP version, where the second channel represents pressure and the third channel
        represents volume from the time series to construct RGB images.

        Each MLF and titration feature set is saved to a file in a new feature directory.
        Files named `Res_<id>.npz` contain features after ResNet extraction, and
        `LSTM_<id>.npz` contain the final features after LSTM processing.
        """

        logger.info("Start feature for: " + self.split)

        torch.set_grad_enabled(False)
        self.resnet.eval()
        self.lstm.eval()

        pat_folders = sorted([f for f in (self.path / self.set / self.split).iterdir() if f.is_dir()])
        for idx, p in enumerate(pat_folders, start=1):
            logger.info(f"Starting iteration for: {p.name} pat: {idx} / {len(pat_folders)}")
            logger.info("3D")
            logger.info("MLF")
            mlfPath = p / "MLFeit.npz"

            if not mlfPath.exists():
                logger.warning(f"Missing file, skipping: {mlfPath}")
                continue

            mlfData = np.load(mlfPath)
            mlfRes3D = self.extract_resnet_features3D(mlfData['EITdata'], "MLF")
            mlfLSTM3D = self.extract_LSTM_features(mlfRes3D).cpu()

            logger.info("TIT")
            titPath = p / "TITeit.npz"
            titData = np.load(titPath)
            titRes3D = self.extract_resnet_features3D(titData['EITdata'], "TIT")
            titLSTM3D = self.extract_LSTM_features(titRes3D).cpu()

            logger.info("save")
            save_path = Path(self.path) / 'feature' / self.set / "3D" / self.split
            save_path.mkdir(parents = True, exist_ok = True)

            np.savez_compressed(
                save_path / f"{p.name}_Res.npz",
                patient_id = p.name,
                mlf = mlfRes3D.cpu().numpy(),
                tit = titRes3D.cpu().numpy(),
            )

            np.savez_compressed(
                save_path / f"{p.name}_LSTM.npz",
                patient_id = p.name,
                mlf = mlfLSTM3D.cpu().numpy(),
                tit = titLSTM3D.cpu().numpy(),
            )
            del mlfLSTM3D, titLSTM3D

            logger.info("vp")
            mlfResVP = self.extract_resnet_featuresVP(
                (mlfData['EITdata']), 
                (mlfData['pressureData']), 
                (mlfData['volumeData']),
                "MLF"
                )
            mlfLSTMVP = self.extract_LSTM_features(mlfResVP).cpu()

            titResVP = self.extract_resnet_featuresVP(
                (titData['EITdata']), 
                (titData['pressureData']), 
                (titData['volumeData']),
                "TIT"
                )
            titLSTMVP = self.extract_LSTM_features(titResVP).cpu()

            logger.info("save")
            save_path = self.path / 'feature' / self.set / "VP" / self.split
            save_path.mkdir(parents = True, exist_ok = True)

            np.savez_compressed(
                save_path / f"{p.name}_Res.npz",
                patient_id = p.name,
                mlf = mlfResVP.cpu().numpy(),
                tit = titResVP.cpu().numpy(),
            )

            np.savez_compressed(
                save_path / f"{p.name}_LSTM.npz",
                patient_id = p.name,
                mlf = mlfLSTMVP.cpu().numpy(),
                tit = titLSTMVP.cpu().numpy(),
            )

            path = p / "table.npz"
            data = np.load(path)

            save_path = self.path / 'feature' / self.set / "table" / self.split
            save_path.mkdir(parents = True, exist_ok = True)


            np.savez_compressed(
                save_path / f"{p.name}.npz",

                sex    = np.array([data['sex']], dtype=np.float32),
                age    = np.array([(data['age'] - self.amean) / (self.astd + 1e-6)], dtype=np.float32),
                BMI    = np.array([(data['BMI'] - self.bmean) / (self.bstd + 1e-6)], dtype=np.float32),
                weight = np.array([(data['weight'] - self.wmean) / (self.wstd + 1e-6)], dtype=np.float32),
                height = np.array([(data['height'] - self.hmean) / (self.hstd + 1e-6)], dtype=np.float32),
                smoke  = np.array([data['smoke']], dtype=np.float32),

                targetPEEP = np.array(
                    [(data['targetPEEP'] - self.pmean) / (self.pstd + 1e-6)],
                    dtype=np.float32
                ),
            )



    def get_tglobal(self) -> Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
        asum = bsum = wsum = hsum = psum = 0.0
        asq_sum = bsq_sum = wsq_sum = hsq_sum = psq_sum = 0.0
        count = 0

        pat_folders = sorted([f for f in (self.path / self.set / self.split).iterdir() if f.is_dir()])
        assert len(pat_folders) != 0, "No patient folders found in the path!"
        for p in pat_folders:
            path = p / "table.npz"
            
            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue

            data = np.load(path)

            asum += data['age'].sum()
            asq_sum += np.square(data['age']).sum()
            bsum += data['BMI'].sum()
            bsq_sum += np.square(data['BMI']).sum()
            wsum += data['weight'].sum()
            wsq_sum += np.square(data['weight']).sum()
            hsum += data['height'].sum()
            hsq_sum += np.square(data['height']).sum()
            psum += data['targetPEEP'].sum()
            psq_sum += np.square(data['targetPEEP']).sum()
            count += 1

        if count == 0:
            raise RuntimeError(f"No data found for table")

        
        amean = asum / count
        var = asq_sum / count - amean**2
        var = np.maximum(var, 0.0)
        astd = np.sqrt(var) + 1e-6
        bmean = bsum / count
        var = bsq_sum / count - bmean**2
        var = np.maximum(var, 0.0)
        bstd = np.sqrt(var) + 1e-6
        wmean = wsum / count
        var = wsq_sum / count - wmean**2
        var = np.maximum(var, 0.0)
        wstd = np.sqrt(var) + 1e-6
        hmean = hsum / count
        var = hsq_sum / count - hmean**2
        var = np.maximum(var, 0.0)
        hstd = np.sqrt(var) + 1e-6
        pmean = psum / count
        var = psq_sum / count - pmean**2
        var = np.maximum(var, 0.0)
        pstd = np.sqrt(var) + 1e-6

        return np.float32(amean), np.float32(astd), np.float32(bmean), np.float32(bstd), np.float32(wmean), np.float32(wstd), np.float32(hmean), np.float32(hstd), np.float32(pmean), np.float32(pstd)
    
    def get_global(self, modality:str) -> Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
        """
        Iterrating over all raw data picture arrays for the longest breath or for titration. 
        Collect total sum, total square sum and count the total number of pixels.
        Returning the per pixel mean and standart diviation
        
        Input:
        modality : str
            dataset identifier ("MLF", "TIT")

        Returns:
        arr_mean : numpy.float32
            global scaler
        arr_std : numpy.float32
            global scalar
        p_mean : numpy.float32
            global scaler
        p_std : numpy.float32
            global scalar
        v_mean : numpy.float32
            global scaler
        v_std : numpy.float32
            global scalar
        """
        logger.info("Start get_global for: " + modality)

        p_sum = v_sum = 0.0
        p_sq_sum = v_sq_sum = 0.0
        p_count = v_count = 0
        amax = -np.inf
        amin = np.inf

        pat_folders = sorted([f for f in (self.path / self.set / self.split).iterdir() if f.is_dir()])
        for p in pat_folders:
            path = p / f"{modality}eit.npz"
            
            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue

            data = np.load(path)

            #arr_sum += np.nansum(data['EITdata'])
            #arr_sq_sum += np.nansum(np.square(data['EITdata']))
            #arr_count += np.count_nonzero(~np.isnan(data['EITdata']))

            amin = min(amin, np.nanmin(data["EITdata"]))
            amax = max(amax, np.nanmax(data["EITdata"]))

            p_sum += data['pressureData'].sum()
            p_sq_sum += np.square(data['pressureData']).sum()
            p_count += data['pressureData'].size
 
            v_sum += data['volumeData'].sum()
            v_sq_sum += np.square(data['volumeData']).sum()
            v_count += data['volumeData'].size 

            del data

        if p_count == 0 or v_count == 0:
            raise RuntimeError(f"No pressure/volume data found for modality {modality}")
        
        #arr_mean = arr_sum / arr_count
        #arr_var = arr_sq_sum / arr_count - arr_mean**2
        #arr_var = np.maximum(arr_var, 0.0)
        #arr_std = np.sqrt(arr_var) + 1e-6 

        p_mean = p_sum / p_count
        p_var = p_sq_sum / p_count - p_mean**2
        p_var = np.maximum(p_var, 0.0)
        p_std = np.sqrt(p_var) + 1e-6

        v_mean = v_sum / v_count
        v_var = v_sq_sum / v_count - v_mean**2
        v_var = np.maximum(v_var, 0.0)
        v_std = np.sqrt(v_var) + 1e-6

        return np.float32(amin), np.float32(amax), np.float32(p_mean), np.float32(p_std), np.float32(v_mean), np.float32(v_std)
    

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

    def extract_resnet_features3D(self, frames:np.ndarray, modality:str, batch_size:int=256) -> torch.Tensor:
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
                

                chunk = chunk.unsqueeze(1)                  # (B,1,H,W)
                chunk = chunk.expand(-1, 3, -1, -1).clone() # (B,3,H,W)

                mask = ~torch.isnan(chunk)
                
                if modality == "MLF":
                    chunk[mask] = (chunk[mask] - self.MLFmin) / (self.MLFmax - self.MLFmin + 1e-6)
                else:
                    chunk[mask] = (chunk[mask] - self.TITmin) / (self.TITmax - self.TITmin + 1e-6)

                chunk[:,0,:,:][mask[:,0,:,:]] = (chunk[:,0,:,:][mask[:,0,:,:]] - 0.485) / 0.229
                chunk[:,1,:,:][mask[:,1,:,:]] = (chunk[:,1,:,:][mask[:,1,:,:]] - 0.456) / 0.224
                chunk[:,2,:,:][mask[:,2,:,:]] = (chunk[:,2,:,:][mask[:,2,:,:]] - 0.406) / 0.225

                chunk[~mask] = 0.0

                chunk = F.interpolate(
                    chunk,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                features = self.resnet(chunk)         # (B,2048,1,1)
                features = features.flatten(1)        # (B,2048)

                features = F.relu(self.resnet_reduce(features))#(B, 256)

                spatial_features.append(features)

        return torch.cat(spatial_features, dim=0)
    

    def extract_resnet_featuresVP(self, frames:np.ndarray, pre:np.ndarray, vol:np.ndarray, modality:str, batch_size=256) -> torch.Tensor:
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
                

                mask = ~torch.isnan(chunk)
                if modality == "MLF":
                    chunk[mask] = (chunk[mask] - self.MLFmin) / (self.MLFmax - self.MLFmin + 1e-6)
                else:
                    chunk[mask] = (chunk[mask] - self.TITmin) / (self.TITmax - self.TITmin + 1e-6)
                chunk[:,0,:,:][mask[:,0,:,:]] = (chunk[:,0,:,:][mask[:,0,:,:]] - 0.485) / 0.229   
                chunk[~mask] = 0.0

                p = torch.from_numpy(pre[0, i:i+batch_size]).float().to(self.device)
                v = torch.from_numpy(vol[0, i:i+batch_size]).float().to(self.device)
                                        
                if modality == "MLF":
                    p = (p - self.MLFpmean) / (self.MLFpstd + 1e-6)
                    v = (v - self.MLFvmean) / (self.MLFvstd + 1e-6)
                else:
                    p = (p - self.TITpmean) / (self.TITpstd + 1e-6)
                    v = (v - self.TITvmean) / (self.TITvstd + 1e-6)

                pChunk = p.view(-1, 1, 1, 1).expand(-1, 1, chunk.shape[2], chunk.shape[3])
                vChunk = v.view(-1, 1, 1, 1).expand(-1, 1, chunk.shape[2], chunk.shape[3])
                vp = torch.cat([chunk, pChunk, vChunk], dim=1)

                vp = F.interpolate(
                    vp,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                features = self.resnet(vp)          # (B,2048,1,1)
                features = features.flatten(1)      # (B,2048)

                features = F.relu(self.resnet_reduce(features))

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
        seq = spatial_seq.unsqueeze(0)              # (1, T, 256)
        out, (h_n, c_n) = self.lstm(seq)

        lstm_features = torch.cat([h_n[-1], out.mean(dim=1), out.max(dim=1).values], dim=1)                         # 3*(1, 128) = (1, 384)
        lstm_features = F.relu(self.lstm_reduce(lstm_features))                                                     # (1,128)  testen mit end ziel 16 - 32
        return lstm_features  