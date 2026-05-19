import math

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


def ci_artifact_reduction(raw, subject_id, trial_id, output_dir, snr_threshold, fs_eeg, attended_audio, distraction_audio=None, peak_win_negative = 0.005, peak_win_pos = 0.012, plot=False, metadata=False):
    """
        Reduce CI Artifacts from EEG data

        Input:
        - raw: mne.io.Raw
#         EEG data loaded from an EEGLAB .set file
        - subject_id: str or int #TODO welches Format muss das haben?
            Identifier for the subject, used for naming output files and saving metadata
        - trial_id: str or int #TODO welches Format muss das haben?
            Identifier for the trial, used for naming output files and saving metadata
        - output_dir : str
            Path where output files will be saved
        - snr_threshold: float 
            Maximum SNR value required for an independent component to not be extracted from the dataset
        - fs_eeg : int
            Sampling frequency of the EEG recording in Hz
        - attended_audio : np.ndarray
            1D NumPy array containing the signal of the attended audio stream

        - distraction_audio : np.ndarray
            1D NumPy array containing the signal of the distracting (unattended) audio stream if available, default is None (for single speaker scenarios)
        - peak_win_negative : float
            Negative time lag window (in seconds), default 0.005s
        - peak_win_positive : float
            Positive time lag window (in seconds), default 0.012s
        - plot : bool
            If True, enables saving of plots, default is False
        - metadata : bool
            If True, enables saving of metadata, default is False

        Returns:
        - eeg_cleaned: EEG data without CI Artifacts  
        """

        #TODO:
        # DONE Check ob EEG und Audio attended/ distracted selbe länge haben
        # DONE Check obs ein single speaker oder conmpeting speaker ist, wenn competing ist audio attended + competing 
        # DONE Kommentare unterdrücken?
        # DONE plot 
        # - subject_id und trial_id einflechten und für richtige Speicherung nutzen
        # - Kommentare schön machen

    #Error if no output path
    if output_dir is None:
        raise ValueError("Output path must be specified.")
    
    #prepare output directory
    if plot == True or metadata == True:
        output_dir = Path(output_dir) / "output_corsica"
        output_dir.mkdir(parents=True, exist_ok=True)

    #prepare audio_sum
    #set competing to true if necessary and adjust audio_sum    
    audio_sum = attended_audio
    if isinstance(distraction_audio, np.ndarray):
            audio_sum = attended_audio + distraction_audio

    #prepare eeg
    # load eeg with all components
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
            "Only tested with frequencies >= 500 Hz. Results may be unreliable.",
            UserWarning
    )

    #iterate over all ic
    exclude = []
    snr_s= []
    lags_s = []
    corr_s = []
    peak_in_seconds_after_stimulus_s = []
    for ic in range(ics.n_components_):
        corr = signal.correlate(ica_data[ic, :], audio_sum)
        corr_s.append(corr)
        corr /= np.max(corr)
        lags = signal.correlation_lags(len(ica_data[ic, :]), len(audio_sum))
        lags_s.append(lags)
        snr, central_lags, peak_value_idx, peak_in_seconds_after_stimulus = peak_snr(corr, fs_eeg, peak_win_negative, peak_win_pos)
        snr_s.append(snr)
        peak_in_seconds_after_stimulus_s.append(peak_in_seconds_after_stimulus)

        # reject components based on snr value
        if snr > snr_threshold: 
            exclude.append(ic)

        
    
    #if plot is true save plot 
    if plot == True:
        plotting(lags_s,corr_s,fs_eeg, snr_s, output_dir, subject_id, trial_id, peak_win_negative, peak_win_pos)

    # reconstruct EEG based on remaining ICs
    ics.exclude = exclude
    raw_cleaned = raw.copy()
    ics.apply(raw_cleaned)

    cleaned_eeg = raw_cleaned.get_data()

    
    
    if metadata == True: 
        calculate_metadata(ics, exclude, snr_s, peak_in_seconds_after_stimulus_s, output_dir, subject_id, trial_id)

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

def plotting(lags_s,corr_s,fs_eeg, snr_s, output_dir, subject_id, trial_id, peak_win_negative, peak_win_pos):
    n = len(lags_s) 

    n_cols = 4
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots (n_rows, n_cols, figsize=(4* n_cols, 3*n_rows))
    axes = axes.flatten()
    
    for i in range (n):
        ax = axes[i]
        lags = lags_s[i]
        corr = corr_s[i]
        snr = snr_s[i]

        # 1. Convert lags to ms (Assuming 'lags' is in seconds, * 1000)
        lags_ms = (lags / 1000) * 1000 #NOTE wieso macht man hier geteilt durch und mal 1000

        # cross-correlation plot
        ax.plot(lags_ms, corr, color='dimgrey', linewidth=1.5)

        # Mark the search window (-5ms to 15ms) in grey
        ax.axvspan(peak_win_negative * 1000, peak_win_pos * 1000, color='grey', alpha=0.3, label='Search Window')

        
        # Highlight Peak within the window
        # Ensure fs_eeg and corr are available in your local scope
        n_samples = corr.shape[0]
        start_idx = n_samples // 2 - int(0.005 * fs_eeg)
        stop_idx = n_samples // 2 + int(0.015 * fs_eeg)
        central_indices = np.arange(start_idx, stop_idx)

        peak_val = np.max(corr[central_indices])
        peak_idx = np.argmax(corr[central_indices]) + start_idx

        ax.plot(lags_ms[peak_idx], peak_val, marker='x', color='black', markersize=8, mew=2)

        #labels
        ax.set_xlim([-300, 300])
        #ax.set_ylim([-1.1, 1.1])
        ax.set_title(f"IC {i+1}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Delay (ms)', fontsize=10)
        ax.set_ylabel('Cross-correlation', fontsize=10)
        #ax.set_title("Correlation-based artifact rejection", fontsize=16, pad=15, weight='bold')

        ax.text(-150, 0.6, 'Search\nwindow', fontsize = 12)
        ax.text(100, 1.0, f'Peak\nSNR={snr:.1f}dB', fontsize=12)

        # Remove upper and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(axis='both', which='major', labelsize=14)
    
    
    # remove empty subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Correlation-based CI artifact reduction – cors-ica", fontsize=18, y=0.98)
    fig.suptitle(f"Cross-Correlation plots of all Independent Components\n"
             f"Subject: {subject_id} | Trial: {trial_id}", 
             fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    #one pdf with everything
    plt.savefig(output_dir / f"{subject_id}_corsica_correlation_ic_trial_{trial_id}.pdf")

    plt.close(fig)

    return 

def calculate_metadata(ics, exclude, snr_s, peak_in_seconds_after_stimulus_s, output_dir, subject_id, trial_id):
    #calculate values
    number_of_ics= ics.n_components_
    number_excluded_ics = len(exclude) 
    percentage_remaining_ics = (number_of_ics - number_excluded_ics) / number_of_ics * 100
    excluded_snr_values = [round(float(snr_s[ic]), 3) for ic in exclude]
    number_used_ics = number_of_ics - number_excluded_ics
    used_snr_indizes = [i for i in range(number_of_ics) if i not in exclude]
    used_snr_values = [round(float(snr_s[i]),3) for i in range(number_of_ics) if i not in exclude]
    max_snr= max(snr_s)
    excluded_peak_times_in_seconds_after_stimulus = [peak_in_seconds_after_stimulus_s[i] for i in exclude]
    mean_peak_in_seconds_after_stimulus_s = np.mean(excluded_peak_times_in_seconds_after_stimulus)
    mean_snr = np.mean(snr_s)
    mean_snr_cleaned = np.mean([snr_s[i] for i in range(number_of_ics) if i not in exclude])
    mean_snr_excluded = np.mean(excluded_snr_values) if excluded_snr_values else float('nan')

    data = {
        "Trial ID": trial_id,
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

    # filename per subject
    output_dir = Path(output_dir)
    csv_file = output_dir / f"{subject_id}_corsica_eeg_metrics.csv"

    # check if file exists
    file_exists = csv_file.exists()

    # append row
    df.to_csv(
        csv_file,
        mode="a",
        header=not file_exists,
        index=False
    )