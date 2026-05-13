import numpy as np
import mne
from CORSICA.artifact_reduction import ci_artifact_reduction

#EEG
eeg_path = '/Users/leonierichter/Documents/2100_Work/2026_Uni_Chair_Senory_in_Neuroengineering/2026.03.27_Code_und_Daten/simple_prepro/102/102_Elbenwald_FL_2.set'
raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
snr_threshold= 12
fs_eeg=1000

#Audio
attended_audio = raw.get_data()[31, :]
print(attended_audio)

cleaned_eeg= ci_artifact_reduction(raw, snr_threshold, fs_eeg, attended_audio, plot=True, metadata=True)