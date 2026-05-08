import numpy as np
import warnings
from pathlib import Path
import csv

import matplotlib 
matplotlib.use('Agg')
import mne
mne.set_log_level('WARNING') #Ausgaben von MNE Python minimieren
from scipy import signal
from mne.preprocessing import ICA


def ci_artifact_reduction(raw, snr_threshold, fs_eeg, attended_audio, distraction_audio=None, peak_win_negative = 0.005, peak_win_pos = 0.012, plot=False, metadata=False):
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
    
    
    # reconstruct EEG based on remaining ICs
    ics.exclude = exclude
    raw_cleaned = raw.copy()
    ics.apply(raw_cleaned)

    cleaned_eeg = raw_cleaned.get_data()

    #if plot is true show or save plot #TODO
    if plot == True:
         plot()
    
    if metadata == True: 

         #values to save
         number_of_ics= ics.n_components_
         print(f"Number of ICs: {number_of_ics}")
         number_excluded_ics = len(exclude) 
         print(f"Excluded ICs: {number_excluded_ics} with indices {exclude}")
         percentage_remaining_ics = (number_of_ics - number_excluded_ics) / number_of_ics * 100
         print (f"Percentage of remaining ICs: {percentage_remaining_ics:.2f}% ")
         excluded_snr_values = [round(float(snrs[ic]), 3) for ic in exclude]
         print(f"Snr values of excluded ICs: {excluded_snr_values}")
         number_used_ics = number_of_ics - number_excluded_ics
         used_snr_indizes = [i for i in range(number_of_ics) if i not in exclude]
         print(f"Used ICs: {number_used_ics} with indices {used_snr_indizes}")
         used_snr_values = [round(float(snrs[i]),3) for i in range(number_of_ics) if i not in exclude]
         print(f"Snr values of used ICs: {used_snr_values}")
         max_snr= max(snrs)
         print(f"Highest SNR values of all ICs: {max_snr:.3f} dB")
         mean_peak_in_seconds_after_stimulus_s = np.mean(peak_in_seconds_after_stimulus_s) #NOTE array ausgeben lassen
         print(f"Peak occurs on average at {mean_peak_in_seconds_after_stimulus_s:.3f} seconds after stimulus onset")
         mean_snr = np.mean(snrs)
         print(f"Mean SNR of all ICs: {mean_snr:.3f} dB")
         mean_snr_cleaned = np.mean([snrs[i] for i in range(number_of_ics) if i not in exclude])
         print(f"Mean SNR of remaining ICs: {mean_snr_cleaned:.3f} dB") #NOTE nicht so viel aussagekraft weil es ja das snr von dem signal ist was als artefakt gilt
         mean_snr_excluded = np.mean(excluded_snr_values) if excluded_snr_values else float('nan')
         print(f"Mean SNR of excluded ICs: {mean_snr_excluded:.3f} dB")

         current_dir = Path(__file__).parent
         file_path = current_dir / "eeg_metrics.csv"

         data = [
         ["Number of independent components", number_of_ics, "count"],
         ["Number of excluded ICs", number_excluded_ics, "count"],
         ["Indices of excluded ICs", exclude, "count"],
         ["Percentage of remaining ICs", round(percentage_remaining_ics, 2), "%"],
         ["Excluded ICs SNR values", ", ".join(map(str, excluded_snr_values)), "dB"],
         ["Number of used ICs", number_used_ics, "count"],
         ["Indices of used ICs", used_snr_indizes, "count"],
         ["Used ICs SNR values", ", ".join(map(str, used_snr_values)), "dB"],
         ["Highest SNR", round(max_snr, 3), "dB"],
         ["Mean SNR", round(mean_snr, 3), "dB"],
         ["Mean peak time in seconds after stimulus onset", round(mean_peak_in_seconds_after_stimulus_s, 3), "s"],
         ["Mean SNR of remaining ICs", round(mean_snr_cleaned, 3), "dB"],
         ["Mean SNR of excluded ICs", round(mean_snr_excluded, 3), "dB"]
         ]
 
         with open(file_path, mode='w', newline='') as f:
             writer = csv.writer(f)
             writer.writerow(["metric", "value", "unit"])
             writer.writerows(data)
            

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
    cross_corr_no_peak = np.delete(correlation, peak_value_idx) #NOTE: man löscht nur eine indez aber ist der Peak nicht auf einer breiteren Spanne 

    noise_power = np.mean(cross_corr_no_peak ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    #calculate when peak occurs in seconds after 0 lag
    peak_in_seconds_after_stimulus = (peak_value_idx - center) / fs


    return snr, central_lags, peak_value_idx, peak_in_seconds_after_stimulus

def check_dimensions(audio, eeg_data):
    # Checks whether the audio and EEG arrays have the same length.
    if len(audio) != eeg_data.shape[1]:
        raise ValueError(f"Dimensions do not match! Audio: {len(audio)}, EEG: {eeg_data.shape[1]}")
    else:
        print(f"Check successful: Arrays have the same length. Audio: {len(audio)}, EEG: {eeg_data.shape[1]}")

def plot():
     return 