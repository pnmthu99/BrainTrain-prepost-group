import mne
from collections import Counter

FILE_PATH = "/mnt/data_lab513/thupnm/BrainTrain-prepost-group/EEG/raweeg_01m/B02_00/B02_00.edf"
TASK_LABELS = {"3A", "3B", "4", "5", "6"}
TASK_DURATION_SEC = 60.0

raw = mne.io.read_raw_edf(FILE_PATH, preload=False, verbose=False)

print(f"Recording duration: {raw.times[-1]:.1f} seconds")
print("\nAll annotations in the EDF:")

task_counts = Counter()

for annot in raw.annotations:
    description = annot["description"]
    onset = annot["onset"]
    duration = annot["duration"]

    # Cùng cách chuẩn hóa marker mà pipeline đang dùng
    marker = description.split("(")[0].strip().upper()

    valid_60s = onset + TASK_DURATION_SEC <= raw.times[-1]
    print(
        f"  onset={onset:9.3f}s | duration={duration:7.3f}s | "
        f"description={description!r} | normalized={marker!r} | "
        f"enough_for_60s={valid_60s}"
    )

    if marker in TASK_LABELS:
        task_counts[marker] += 1

print("\nRecognized task-marker counts:")
for marker in ["3A", "3B", "4", "5", "6"]:
    print(f"  {marker}: {task_counts[marker]}")