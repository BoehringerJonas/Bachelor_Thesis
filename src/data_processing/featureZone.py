import logging
import numpy as np
from pathlib import Path
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Tuple

logger = logging.getLogger("workflow")


class FeatureZone:
    def __init__(self, path: Path, split: str, set: str, device) -> None:
        """
        Defines the different encoders and linear layers and set's global values.

        Parameters
        ----------
        path: Path
            Path to input directory
        split: str
            name of data split (train, val, test)
        set: str
            name of tit split (full_tit, half_tit, third_tit)
        device
        """
        self.path = path
        self.split = split
        self.set = set

        if self.set == "full_tit":
            self.folder = "clean"
        else:
            self.folder = self.set

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("def ResNet")

        self.resnet = self.get_res().to(self.device)
        self.resnet_reduce = nn.Linear(2048, 512).to(self.device)

        self.lstm_MLF = nn.LSTM(
            input_size=512,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(self.device)

        self.lstm_TIT = nn.LSTM(
            input_size=512,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(self.device)

        self.lstm_reduce_MLF = nn.Linear(192, 16).to(self.device)
        self.lstm_reduce_TIT = nn.Linear(192, 16).to(self.device)

        self.resnet.eval()
        self.lstm_MLF.eval()
        self.lstm_TIT.eval()

    def run_subset(self, pat_folders:Path) -> None:
        """
        Starts featuring prozess per patient.

        Parameters
        ----------
        pat_folders:Path
            path to given input directorys
        """
        self.load_stats()
        for p in pat_folders:
            self.process_patient(p)

    def process_patient(self, p: Path) -> None:
        """
        Loads patient files and extract features with a Resnet50 and a LSTM encoder out 
        of MLF and TIT data. 
        Normalize and safe tabular data.
        
        Parameters
        ----------
        p: Path
            Path to spezified patient
        """
        logger.info(f"Starting iteration for: {p.name}")
        logger.info("3D")

        mlfPath = p / "MLFeit.npz"
        if not mlfPath.exists():
            logger.warning(f"Missing file, skipping: {mlfPath}")
            return

        mlfData = np.load(mlfPath)

        mlfRes3D = self.extract_resnet_features3D(mlfData['EITdata'], "MLF").detach().cpu()
        mlfLSTM3D = self.extract_LSTM_features(mlfRes3D, "MLF").detach().cpu()


        titPath = p / "TITeit.npz"
        titData = np.load(titPath)

        titRes3D = self.extract_resnet_features3D(titData['EITdata'], "TIT").detach().cpu()
        titLSTM3D = self.extract_LSTM_features(titRes3D, "TIT").detach().cpu()

        save_path = self.path / 'feature' / self.set / "3D" / self.split
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path / f"{p.name}_Res.npz",
            patient_id=p.name,
            mlf=mlfRes3D.numpy(),
            tit=titRes3D.numpy(),
        )

        np.savez_compressed(
            save_path / f"{p.name}_LSTM.npz",
            patient_id=p.name,
            mlf=mlfLSTM3D.numpy(),
            tit=titLSTM3D.numpy(),
        )

        del mlfLSTM3D, titLSTM3D

        logger.info("VP")

        mlfResVP = self.extract_resnet_featuresVP(
            mlfData['EITdata'],
            mlfData['pressureData'],
            mlfData['volumeData'],
            "MLF"
        ).detach().cpu()

        mlfLSTMVP = self.extract_LSTM_features(mlfResVP, "MLF").detach().cpu()

        titResVP = self.extract_resnet_featuresVP(
            titData['EITdata'],
            titData['pressureData'],
            titData['volumeData'],
            "TIT"
        ).detach().cpu()

        titLSTMVP = self.extract_LSTM_features(titResVP, "TIT").detach().cpu()

        save_path = self.path / 'feature' / self.set / "VP" / self.split
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path / f"{p.name}_Res.npz",
            patient_id=p.name,
            mlf=mlfResVP.numpy(),
            tit=titResVP.numpy(),
        )

        np.savez_compressed(
            save_path / f"{p.name}_LSTM.npz",
            patient_id=p.name,
            mlf=mlfLSTMVP.numpy(),
            tit=titLSTMVP.numpy(),
        )

        path = p / "table.npz"
        data = np.load(path)

        save_path = self.path / 'feature' / self.set / "table" / self.split
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path / f"{p.name}.npz",
            sex=np.array([data['sex']], dtype=np.float32),
            age=np.array([(data['age'] - self.amean) / self.astd], dtype=np.float32),
            BMI=np.array([(data['BMI'] - self.bmean) / self.bstd], dtype=np.float32),
            weight=np.array([(data['weight'] - self.wmean) / self.wstd], dtype=np.float32),
            height=np.array([(data['height'] - self.hmean) / self.hstd], dtype=np.float32),
            smoke=np.array([data['smoke']], dtype=np.float32),
            targetPEEP=np.array(
                [(data['targetPEEP'] - self.pmean) / self.pstd],
                dtype=np.float32
            ),
        )
        logger.info(f"Finish Patient: {p.name}")

    def calc_stats(self):
        """
        Coordinates the calculation of min, max, mean and std of the data. Safes everything.
        """
        if self.split != "train":
            raise RuntimeError(
                f"calc_stats darf nur auf dem train-Split laufen, nicht auf '{self.split}'"
            )
            
        MLFmean, MLFstd, MLFpmean, MLFpstd, MLFvmean, MLFvstd = self.get_global("MLF")
        TITmean, TITstd, TITpmean, TITpstd, TITvmean, TITvstd = self.get_global("TIT")

        amean, astd, bmean, bstd, wmean, wstd, hmean, hstd, pmean, pstd = self.get_tglobal()

        spath = self.path / "feature" / self.set / "stats"
        spath.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            spath / "train_stats.npz",
            MLFmean=MLFmean,
            MLFstd=MLFstd,
            MLFpmean=MLFpmean,
            MLFpstd=MLFpstd,
            MLFvmean=MLFvmean,
            MLFvstd=MLFvstd,
            TITmean=TITmean,
            TITstd=TITstd,
            TITpmean=TITpmean,
            TITpstd=TITpstd,
            TITvmean=TITvmean,
            TITvstd=TITvstd,
            amean=amean,
            astd=astd,
            bmean=bmean,
            bstd=bstd,
            wmean=wmean,
            wstd=wstd,
            hmean=hmean,
            hstd=hstd,
            pmean=pmean,
            pstd=pstd,
        )

    def load_stats(self):
        """
        Loads min, max, mean and std of the data calculated in calc_stats to global variable.
        """
        spath = self.path / "feature" / self.set / "stats"
        stats = np.load(spath / "train_stats.npz")

        self.MLFmean = torch.tensor(stats["MLFmean"], device=self.device)
        self.MLFstd = torch.tensor(stats["MLFstd"], device=self.device)
        self.TITmean = torch.tensor(stats["TITmean"], device=self.device)
        self.TITstd = torch.tensor(stats["TITstd"], device=self.device)

        self.MLFpmean = torch.tensor(stats["MLFpmean"], device=self.device)
        self.MLFpstd = torch.tensor(stats["MLFpstd"], device=self.device)
        self.MLFvmean = torch.tensor(stats["MLFvmean"], device=self.device)
        self.MLFvstd = torch.tensor(stats["MLFvstd"], device=self.device)

        self.TITpmean = torch.tensor(stats["TITpmean"], device=self.device)
        self.TITpstd = torch.tensor(stats["TITpstd"], device=self.device)
        self.TITvmean = torch.tensor(stats["TITvmean"], device=self.device)
        self.TITvstd = torch.tensor(stats["TITvstd"], device=self.device)

        self.amean = stats["amean"]
        self.astd = stats["astd"]
        self.bmean = stats["bmean"]
        self.bstd = stats["bstd"]
        self.wmean = stats["wmean"]
        self.wstd = stats["wstd"]
        self.hmean = stats["hmean"]
        self.hstd = stats["hstd"]
        self.pmean = stats["pmean"]
        self.pstd = stats["pstd"]

    def get_tglobal(self) -> Tuple[
        np.float32, np.float32,
        np.float32, np.float32,
        np.float32, np.float32,
        np.float32, np.float32,
        np.float32, np.float32
    ]:
        """
        Calculates mean and std for the tabular data.

        Returns
        -------
        np.float32 : mean age, np.float32 : std age,
        np.float32 : mean bmi, np.float32 : std bmi,
        np.float32 : mean weight, np.float32 : std weight,
        np.float32 : mean hight, np.float32 : std hight,
        np.float32 : mean PEEP, np.float32 : std PEEP
        """
        asum = bsum = wsum = hsum = psum = 0.0
        asq_sum = bsq_sum = wsq_sum = hsq_sum = psq_sum = 0.0
        count = 0

        pat_folders = sorted([
            f for f in (self.path / self.folder / self.split).iterdir()
            if f.is_dir()
        ])

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
            raise RuntimeError("No data found for table")

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

        return (
            np.float32(amean), np.float32(astd),
            np.float32(bmean), np.float32(bstd),
            np.float32(wmean), np.float32(wstd),
            np.float32(hmean), np.float32(hstd),
            np.float32(pmean), np.float32(pstd)
        )

    def get_global(self, modality:str) -> Tuple[
        np.float32, np.float32,
        np.float32, np.float32,
        np.float32, np.float32
    ]:
        """
        Calculates min max for picture data and mean and std for pressure and volume data.

        Parameters
        ----------
        modality: str
            Sets the modality of TIT or MLF to load the correct file.

        Returns
        -------
        np.float32 : min, np.float32 : max,
        np.float32 : mean of pressure, np.float32 : std of pressure,
        np.float32 : mean of volume, np.float32 std of volume
        """
        logger.info("Start get_global for: " + modality)

        p_sum = v_sum = a_sum = 0.0
        p_sq_sum = v_sq_sum = a_sq_sum = 0.0
        p_count = v_count = a_count = 0

        pat_folders = sorted([
            f for f in (self.path / self.folder / self.split).iterdir()
            if f.is_dir()
        ])

        for p in pat_folders:
            path = p / f"{modality}eit.npz"

            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue

            data = np.load(path)

            a_sum += data['EITdata'].sum()
            a_sq_sum += np.square(data['EITdata']).sum()
            a_count += data['EITdata'].size

            p_sum += data['pressureData'].sum()
            p_sq_sum += np.square(data['pressureData']).sum()
            p_count += data['pressureData'].size

            v_sum += data['volumeData'].sum()
            v_sq_sum += np.square(data['volumeData']).sum()
            v_count += data['volumeData'].size

            del data
        if p_count == 0 or v_count == 0:
            raise RuntimeError(f"No pressure/volume data found for modality {modality}")
        
        a_mean = a_sum / a_count
        a_var = a_sq_sum / a_count - a_mean**2
        a_var = np.maximum(a_var, 0.0)
        a_std = np.sqrt(a_var) + 1e-6

        p_mean = p_sum / p_count
        p_var = p_sq_sum / p_count - p_mean**2
        p_var = np.maximum(p_var, 0.0)
        p_std = np.sqrt(p_var) + 1e-6

        v_mean = v_sum / v_count
        v_var = v_sq_sum / v_count - v_mean**2
        v_var = np.maximum(v_var, 0.0)
        v_std = np.sqrt(v_var) + 1e-6

        return (
            np.float32(a_mean), np.float32(a_std),
            np.float32(p_mean), np.float32(p_std),
            np.float32(v_mean), np.float32(v_std)
        )

    def get_res(self) -> nn.Module:
        """
        Defines REsnet 50 Model with the spezified weights.

        Returns
        -------
        nn.Module : REsnet 50 Model
        """
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT) # Try different weights
        resnet = nn.Sequential(*list(resnet.children())[:-1])

        for p in resnet.parameters():
            p.requires_grad = False

        return resnet


    def extract_resnet_features3D(self, frames:np.ndarray, modality:str, batch_size=2048) -> torch.Tensor:
        """
        Builds 3D pictures out of the gray scale pictures and normalized. Set NaN to 0.0 and bring 
        them to the correct form for resnet. Extract features with the resnet encoder and reduce 
        to needet size.

        Parameters
        ----------
        frames      : np.ndarray
            EIT picture Data.
        modality    : str
            MLF or TIT to load correct stats.
        batch_size  : int
            Size of prozessable data at ones.

        Returns
        -------
        torch.Tensor: Features from Resnet
        """
        spatial_features = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in range(0, frames.shape[0], batch_size):
                chunk = torch.from_numpy(frames[i:i + batch_size]).float().to(self.device)
                chunk = chunk.unsqueeze(1).repeat(1, 3, 1, 1)

                mask = ~torch.isnan(chunk)

                if modality == "MLF":
                    chunk[mask] = (
                        (chunk[mask] - self.MLFmean)
                        / (self.MLFstd + 1e-6)
                    )
                else:
                    chunk[mask] = (
                        (chunk[mask] - self.TITmean)
                        / (self.TITstd + 1e-6)
                    )

                chunk[~mask] = 0.0

                chunk = F.interpolate(chunk, 
                                      size=(224, 224), 
                                      mode="bilinear", 
                                      align_corners=False)

                features = self.resnet(chunk).flatten(1)
                if torch.isnan(features).any():
                    print("NaNs detected after ResNet 3D!")
                features = F.relu(self.resnet_reduce(features))

                spatial_features.append(features)

        return torch.cat(spatial_features, dim=0)
    
    def extract_resnet_featuresVP(self,
        frames: np.ndarray, pre: np.ndarray,
        vol: np.ndarray, modality: str, 
        batch_size=2048
    ) -> torch.Tensor:
        """
        Builds 3D pictures out of the gray scale pictures, the pressure an volume data and 
        normalized. Set NaN to 0.0 and bring them to the correct form for resnet. Extract 
        features with the resnet encoder and reduce to needet size.

        Parameters
        ----------
        frames      : np.ndarray
            EIT picture Data.
        pre         : np.ndarray
            pressure data.
        vol         : np.ndarray
            volume data
        modality    : str
            MLF or TIT to load correct stats.
        batch_size  : int
            Size of prozessable data at ones.

        Returns
        -------
        torch.Tensor: Features from Resnet
        """
        assert frames.ndim == 3, f"Expected (T,H,W), got {frames.shape}"
        assert pre.ndim == 2, f"Expected (1, P), got {pre.shape}"
        assert vol.ndim == 2, f"Expected (1, V), got {vol.shape}"
        assert frames.shape[0] == pre.shape[1] == vol.shape[1], \
            "Frames, Volume, Pressure do not have the same length"

        spatial_features = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in range(0, frames.shape[0], batch_size):

                chunk = (
                    torch.from_numpy(frames[i:i+batch_size])
                    .float()
                    .to(self.device, non_blocking=True)
                    .unsqueeze(1)
                )  # (B,1,H,W)

                mask = ~torch.isnan(chunk)

                if modality == "MLF":
                    chunk[mask] = (
                        (chunk[mask] - self.MLFmean)
                        / (self.MLFstd + 1e-6)
                    )
                else:
                    chunk[mask] = (
                        (chunk[mask] - self.TITmean)
                        / (self.TITstd + 1e-6)
                    )

                chunk[:, 0, :, :][mask[:, 0, :, :]] = (
                    chunk[:, 0, :, :][mask[:, 0, :, :]]
                )
                chunk[~mask] = 0.0

                p = torch.from_numpy(pre[0, i:i+batch_size]).float().to(self.device)
                v = torch.from_numpy(vol[0, i:i+batch_size]).float().to(self.device)

                if modality == "MLF":
                    p = (p - self.MLFpmean) / self.MLFpstd
                    v = (v - self.MLFvmean) / self.MLFvstd
                else:
                    p = (p - self.TITpmean) / self.TITpstd
                    v = (v - self.TITvmean) / self.TITvstd

                pChunk = p.view(-1, 1, 1, 1).expand(-1, 1, chunk.shape[2], chunk.shape[3])
                vChunk = v.view(-1, 1, 1, 1).expand(-1, 1, chunk.shape[2], chunk.shape[3])

                vp = torch.cat([chunk, pChunk, vChunk], dim=1)

                vp = F.interpolate(
                    vp,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                features = self.resnet(vp)
                if torch.isnan(features).any():
                    print("NaNs detected after ResNetVP!")
                features = features.flatten(1)

                features = F.relu(self.resnet_reduce(features))

                spatial_features.append(features)

        return torch.cat(spatial_features, dim=0)

    def extract_LSTM_features(self, spatial_seq:torch.Tensor, modality:str) -> torch.Tensor:
        """
        Extracts LSTM features with mean and max pooling and reduce to needet size.

        Parameters
        ----------
        spatial_seq : torch.Tensor
            input Resnet features
        modality    : str
            MLF or TIT for correct stats

        Returns
        -------
        torch.Tensor : LSTM features
        """
        seq = spatial_seq.to(self.device, dtype=torch.float32).unsqueeze(0)
        if modality == "MLF":
            out, (h_n, _) = self.lstm_MLF(seq)
            x = torch.cat([h_n[-1], out.mean(1), out.max(1).values], dim=1)
            return F.relu(self.lstm_reduce_MLF(x))
        else:
            out, (h_n, _) = self.lstm_TIT(seq)
            x = torch.cat([h_n[-1], out.mean(1), out.max(1).values], dim=1)
            return F.relu(self.lstm_reduce_TIT(x))


def worker(gpu_id:int, gpu_tasks: List[List[Tuple[Path, str, str, List[Path]]]]) -> None:
    """
    Is started by the workflow and runs the featuring in chucks for the given GPU.
    Uses new featurezone for each possible combination of split and set.

    Parameters
    ----------
    gpu_id      :int 
        Id of the usable GPU.
    gpu_tasks   : List[List[Tuple[
                        Path,   -> Base input path to the data
                        str,    -> set name (full_tit, half_tit, third_tit)
                        str,    -> split name (train, test, val)
                        List[Path] -> pat_folders
                        ]]]
    """
    torch.cuda.set_device(gpu_id)

    tasks = gpu_tasks[gpu_id]
    fz_cache = {}

    for path, split, set_name, pat_folders in tasks:
        key = (split, set_name)

        if key not in fz_cache:
            fz_cache[key] = FeatureZone(
                path,
                split,
                set_name,
                device=torch.device(f"cuda:{gpu_id}")
            )

        fz = fz_cache[key]
        fz.run_subset(pat_folders)