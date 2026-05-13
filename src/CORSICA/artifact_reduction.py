import numpy as np
import warnings
from pathlib import Path
import csv
import pandas as pd

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
mne.set_log_level('WARNING') #Ausgaben von MNE Python minimieren
from scipy import signal
from mne.preprocessing import ICA
from datetime import datetime


def ci_artifact_reduction(raw, snr_threshold, fs_eeg, attended_audio, distraction_audio=None, peak_win_negative = 0.005, peak_win_pos = 0.012, output_dir= "output", plot=False, metadata=False):
    """
        Reduce CI Artifacts from EEG data

        Input:
        - raw: mne.io.Raw
#         EEG data loaded from an EEGLAB .set file
        - snr_threshold: float 
            Maximum SNR value required for an independent component to not be extracted from the dataset
        - fs_eeg : int
            Sampling frequency of the EEG recording in Hz
        - attended_audio : np.ndarray
            1D NumPy array containing the signal of the attended audio stream
        - distraction_audio : np.ndarray
            1D NumPy array containing the signal of the distracting (unattended) audio stream if available otherwise NaN
        - peak_win_negative : float
            Negative time lag window (in seconds), default 0.005s
        - peak_win_positive : float
            Positive time lag window (in seconds), default 0.012s
        - output_dir : str
            Directory where output files will be saved
        - plot : bool
            If True, enables visualization of results, default is False

        Returns:
        - eeg_cleaned: EEG data without CI Artifacts  
        """

        #TODO:
        # DONE Check ob EEG und Audio attended/ distracted selbe länge haben
        # DONE Check obs ein single speaker oder conmpeting speaker ist, wenn competing ist audio attended + competing 
        # DONE Kommentare unterdrücken?
        # - plot 
    #prepare output directory
    if plot == True or metadata == True:
        output_dir = Path(output_dir) / "output_corsica"
        output_dir.mkdir(parents=True, exist_ok=True)

    #prepare audio_sum
    #set competing to true if necessary and adjust audio_sum    
    competing = False #brauche ich diese Variable überhaupt wenn ich audio_sum sowieso direkt berechne? 
    audio_sum = attended_audio
    if isinstance(distraction_audio, np.ndarray):
            competing = True
            audio_sum = attended_audio + distraction_audio

    #prepare eeg
    # load eeg with all components
     #übergeben
    # load ica
    rank = np.linalg.matrix_rank(raw.get_data())
    print("rank:", rank)
    ics = ICA(n_components=rank, method='infomax', random_state=97)
    ics.fit(raw)
    ica_sources = ics.get_sources(raw)
    ica_data, ica_times = ica_sources.get_data(return_times=True) 
    print("n_channels:", len(raw.ch_names))
    print("data shape:", raw.get_data().shape)
    print("rank:", np.linalg.matrix_rank(raw.get_data()))
    print("ICA components:", ics.n_components_)

    #check if audio and eeg have same dimensions
    eeg_data=raw.get_data()
    check_dimensions(audio_sum, eeg_data)

    #check sampling frequency
    if fs_eeg < 500:
        warnings.warn(
            f"Sampling frequency is very low (fs = {fs_eeg} Hz). "
            "Results may be unreliable due to ...?",
            UserWarning
    )

    #iterate over all ic
    exclude = []
    snrs= []
    peak_in_seconds_after_stimulus_s = []
    for ic in range(ics.n_components_):
        corr = signal.correlate(ica_data[ic, :], audio_sum)
        corr /= np.max(corr)
        lags = signal.correlation_lags(len(ica_data[ic, :]), len(audio_sum))
        snr, central_lags, peak_value_idx, peak_in_seconds_after_stimulus = peak_snr(corr, fs_eeg, peak_win_negative, peak_win_pos)
        snrs.append(snr)
        peak_in_seconds_after_stimulus_s.append(peak_in_seconds_after_stimulus)

        # reject components based on snr value
        if snr > snr_threshold: #wichtig
            exclude.append(ic)

        #if plot is true save plot 
        if plot == True:
            plotting(lags,corr,fs_eeg, snr, output_dir)
    
    
    # reconstruct EEG based on remaining ICs
    ics.exclude = exclude
    raw_cleaned = raw.copy()
    ics.apply(raw_cleaned)

    cleaned_eeg = raw_cleaned.get_data()

    
    
    if metadata == True: 
        calculate_metadata(ics, exclude, snrs, peak_in_seconds_after_stimulus_s, output_dir)

    print ('ganze Methode durchgelaufen')

         
    return cleaned_eeg

def peak_snr(correlation, fs, peak_win_negative, peak_win_pos): 
    """ 
    Calculates the signal-to-noise ratio (SNR) of the highest peak
    in relevant time lags (+/- 5ms)

    Input:
    - correlation: cross-correlation array
    - fs: sampling frequency in Hz

    Output:
    - snr: peak SNR in dB
    - central_lags: indices of the search window around lag = 0

    """
    n = len(correlation)
    center = n//2

    start = center - int(peak_win_negative * fs)
    stop = center+ int(peak_win_pos * fs)

    segment = correlation[start:stop]
    central_lags= np.arange(start, stop)

    peak_value = np.max(segment)
    peak_value_idx= np.argmax(segment) + start

    signal_power = peak_value ** 2
    cross_corr_no_peak = np.delete(correlation, peak_value_idx) 

    noise_power = np.mean(cross_corr_no_peak ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    #calculate when peak occurs in seconds after 0 lag
    peak_in_seconds_after_stimulus = float((peak_value_idx - center) / fs)


    return snr, central_lags, peak_value_idx, peak_in_seconds_after_stimulus

def check_dimensions(audio, eeg_data):
    # Checks whether the audio and EEG arrays have the same length.
    if len(audio) != eeg_data.shape[1]:
        raise ValueError(f"Dimensions do not match! Audio: {len(audio)}, EEG: {eeg_data.shape[1]}")
    else:
        print(f"Check successful: Arrays have the same length. Audio: {len(audio)}, EEG: {eeg_data.shape[1]}")

def plotting(lags,corr,fs_eeg, snr, output_dir):
    
    #create plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    # 1. Convert lags to ms (Assuming 'lags' is in seconds, * 1000)
    lags_ms = (lags / 1000) * 1000 

    # 2. Main cross-correlation plot
    ax.plot(lags_ms, corr, color='dimgrey', linewidth=1.5)

    # 3. Mark the search window (-5ms to 15ms) in grey
    ax.axvspan(-5, 15, color='grey', alpha=0.3, label='Search Window')

    # 4. Highlight Peak within the window
    # Ensure fs_eeg and corr are available in your local scope
    n_samples = corr.shape[0]
    start_idx = n_samples // 2 - int(0.005 * fs_eeg)
    stop_idx = n_samples // 2 + int(0.015 * fs_eeg)
    central_indices = np.arange(start_idx, stop_idx)

    peak_val = np.max(corr[central_indices])
    peak_idx = np.argmax(corr[central_indices]) + start_idx

    ax.plot(lags_ms[peak_idx], peak_val, marker='x', color='black', markersize=8, mew=2)

    # 5. Aesthetics: Limits, Labels, and Spines
    ax.set_xlim([-300, 300])
    ax.set_xlabel('Delay (ms)', fontsize=18)
    ax.set_ylabel('Cross-correlation', fontsize=18)
    ax.set_title("Correlation-based artifact rejection", fontsize=16, pad=15, weight='bold')

    ax.text(-200, 0.5, 'Search\nwindow', fontsize = 18)
    ax.text(100, 0.5, f'Peak\nSNR={snr:.1f}dB', fontsize=18)

    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    
    plt.savefig(output_dir / f"test_plotting.svg")
    #fig.savefig(os.path.join(output_path, 'TRF_corr_approach_search', f'method_1_calculation_{str(sub_id)}_{str(wav_name) + str(ic)}_{str(snr)}.svg')) #TODO
    plt.show()

    return 

def calculate_metadata(ics, exclude, snrs, peak_in_seconds_after_stimulus_s, output_dir):
    #values to save
    number_of_ics= ics.n_components_
    number_excluded_ics = len(exclude) 
    percentage_remaining_ics = (number_of_ics - number_excluded_ics) / number_of_ics * 100
    excluded_snr_values = [round(float(snrs[ic]), 3) for ic in exclude]
    number_used_ics = number_of_ics - number_excluded_ics
    used_snr_indizes = [i for i in range(number_of_ics) if i not in exclude]
    used_snr_values = [round(float(snrs[i]),3) for i in range(number_of_ics) if i not in exclude]
    max_snr= max(snrs)
    excluded_peak_times_in_seconds_after_stimulus = [peak_in_seconds_after_stimulus_s[i] for i in exclude]
    mean_peak_in_seconds_after_stimulus_s = np.mean(excluded_peak_times_in_seconds_after_stimulus)
    mean_snr = np.mean(snrs)
    mean_snr_cleaned = np.mean([snrs[i] for i in range(number_of_ics) if i not in exclude])
    mean_snr_excluded = np.mean(excluded_snr_values) if excluded_snr_values else float('nan')

    data = {
        "Number of independent components": number_of_ics,
        "Number of excluded ICs": number_excluded_ics,
        "Indices of excluded ICs": exclude,
        "Percentage of remaining ICs": round(percentage_remaining_ics, 2),
        "Excluded ICs SNR values": excluded_snr_values,
        "Number of used ICs": number_used_ics,
        "Indices of used ICs": used_snr_indizes,
        "Used ICs SNR values": used_snr_values,
        "Highest SNR": round(max_snr, 3),
        "Mean SNR": round(mean_snr, 3),
        "Mean peak time in seconds after stimulus of excluded ICs": round(mean_peak_in_seconds_after_stimulus_s, 5),
        "All peak times in seconds after stimulus": peak_in_seconds_after_stimulus_s,
        "Excluded peak times in seconds after stimulus": excluded_peak_times_in_seconds_after_stimulus,
        "Mean SNR of remaining ICs": round(mean_snr_cleaned, 3),
        "Mean SNR of excluded ICs": round(mean_snr_excluded, 3),
    }

    df = pd.DataFrame([data])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(output_dir / f"eeg_metrics_{timestamp}.csv", index=False)