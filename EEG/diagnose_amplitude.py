"""
Chẩn đoán nhanh: xem biên độ thực tế của epoch đã lưu, so với ngưỡng QC.
Chạy: python3 diagnose_amplitude.py ./epochs/B08_pre-epo.fif
"""
import sys
import numpy as np
import mne

path = sys.argv[1] if len(sys.argv) > 1 else "./epochs/B08_pre-epo.fif"
epochs = mne.read_epochs(path, verbose=False)
data = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_samples)

print(f"Số epoch: {data.shape[0]}, số kênh: {data.shape[1]}")
print()

for i in range(data.shape[0]):
    ptp_per_channel_uv = np.ptp(data[i], axis=1) * 1e6
    print(f"Epoch {i}: ptp min={ptp_per_channel_uv.min():.1f}µV, "
          f"max={ptp_per_channel_uv.max():.1f}µV, "
          f"median={np.median(ptp_per_channel_uv):.1f}µV")

print()
print("Threshold QC: REJECT_PTP_UV = 150µV")
