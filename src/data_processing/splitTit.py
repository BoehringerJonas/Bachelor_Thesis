import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

signal = np.load(r"D:\BA\workspace\data\clean\train\CPAP-006-7\TITeit.npz")

sig = signal["pressureData"][0,:]
step_point = []
peaks, _ = find_peaks(-sig, height=-27, prominence=5)

for i in range(len(peaks)-1):
    if sig[peaks[i]] - sig[peaks[i+1]] >= 1:
        step_point.append(peaks[i])

plt.plot(sig)
plt.plot(peaks, sig[peaks], "rx")
plt.show()

plt.plot(sig)
plt.plot(step_point, sig[step_point], "rx")
plt.show()