import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import time
from scipy.signal import butter, lfilter
import neurokit2 as nk
from ecgdetectors import Detectors
import pandas as pd
import heartpy as hp
from heartpy.datautils import rolling_mean
from scipy.interpolate import interp1d
import wfdb
from wfdb import processing
import scipy.signal as sig
import math
import scipy.signal as scs
import peakutils
from numpy import genfromtxt
from scipy.signal import find_peaks
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import time
from scipy.signal import butter, lfilter
import neurokit2 as nk
from ecgdetectors import Detectors
import pandas as pd
import heartpy as hp
from heartpy.datautils import rolling_mean
from scipy.interpolate import interp1d
import wfdb
from wfdb import processing
import scipy.signal as sig
import math
import scipy.signal as scs
import peakutils
from scipy.signal import find_peaks
import math

def process_ecg_file(filename):
    daty = genfromtxt(filename, delimiter=',')

    # Thresholding sampling frequency distribution to remove possible infinities
    init=1
    thresh_samp_freq=175000
    threshold=0
    explode=daty.shape[0]
    ecg_ref_data=daty[init:explode,1]
    ecg=ecg_ref_data
    timer=daty[1:(explode-init)+1,2]
    inst_fs=1/np.diff(timer)
    inst_fs[inst_fs > thresh_samp_freq] = threshold

    # Appropiate Windowing for most narrow sampling frequency distribution
    segment_length=3000
    sub_segment_length=1500
    err_record=[]

    for i in range(len(inst_fs)-segment_length+1):
        temp_inst_fs=inst_fs[i:i+segment_length]
        err_record.append(np.std(temp_inst_fs))

    val, idx = min((val, idx) for (idx, val) in enumerate(err_record))
    init=idx
    explode=idx+segment_length
    ecg_ref_data=daty[init:explode,1]
    ecg=ecg_ref_data

    #Appropiate sub-windowing to find least perturbed segement
    err_record=[]
    for j in range(len(ecg)-sub_segment_length+1):
        temp_ecg=ecg[j:j+sub_segment_length]
        rlocser,_ = find_peaks(temp_ecg, distance=70,height=0.5)
        n_locs_old=len(rlocser)
        length=11
        order=2
        smoothed_temp_ecg=savgol_filter(temp_ecg,length, order)
        rlocser,_ = find_peaks(smoothed_temp_ecg, distance=70,height=0.5)
        n_locs_new=len(rlocser)
        err_record.append(abs(n_locs_new-n_locs_old))

    val, idx = min((val, idx) for (idx, val) in enumerate(err_record))
    init=idx
    explode=idx+sub_segment_length
    ecg_ref_data_f=ecg[init:explode]
    ecg=ecg_ref_data_f

    timer=daty[1:(explode-init)+1,2]
    inst_fs=1/np.diff(timer)
    inst_fs[inst_fs > thresh_samp_freq] = threshold

    fs_mean=np.mean(inst_fs)
    fs_std=np.std(inst_fs)
    fs = (np.mean(inst_fs)*6)
    length=11
    order=7

    detrended_ecg=signal.detrend(ecg)
    denoised_detrended_ecg=savgol_filter(detrended_ecg,length, order)
    
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    b, a = butter_highpass(14/60, 10, order=2)
    final_ecg = signal.filtfilt(b, a, denoised_detrended_ecg)

    iter=1000
    init=1
    bp=[]
    fser=[]
    for i in range(iter):
        n=np.random.rand(1)[0]
        escape=50
        lamb=2
        amp=200
        maxi=amp+escape
        mini=amp-lamb*escape
        fs = mini+n*(maxi-mini)
        wd, m = hp.process(final_ecg, sample_rate = fs)
        l=m['breathingrate']*60
        if not math.isnan(l):
            bp.append(m['breathingrate']*60)
            fser.append(fs)
            
    return np.array(bp), np.array(fser)

# List of your CSV files
csv_files = [
    "My_new_ECG_data_1_85bpm_8.0601rpm.csv",
    "My_new_ECG_data_2_91bpm_9.305rpm.csv",  # add your other 4 files here
    "My_new_ECG_data_3_86bpm_7.42rpm.csv",
    "My_new_ECG_data_4_94bpm_9.61rpm.csv",
    "My_new_ECG_data_5_88bpm_9.91rpm.csv"
]

# Process each file and store results
all_bpm = []
all_fs = []

for file in csv_files:
    print(f"\nProcessing file: {file}")
    bpm, fs = process_ecg_file(file)
    all_bpm.append(bpm)
    all_fs.append(fs)
    print(f"Average RPM for {file}: {np.mean(bpm)}")
    print(f"Average FS for {file}: {np.mean(fs)}")

# Display all results
print("\nFinal Results:")
print("\nBPM arrays:")
for i, bpm in enumerate(all_bpm):
    print(f"File {i+1} ({csv_files[i]}): Mean BPM = {np.mean(bpm):.2f}")

print("\nFS arrays:")
for i, fs in enumerate(all_fs):
    print(f"File {i+1} ({csv_files[i]}): Mean FS = {np.mean(fs):.2f}")

