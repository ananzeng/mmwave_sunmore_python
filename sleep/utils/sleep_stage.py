import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import scipy
import datetime
import pandas as pd
from tqdm import tqdm
import csv
import scipy.signal
from scipy.fftpack import fft
import seaborn as sns
# import util
from utils.util import *
from utils.brhr_function import *

def bmi(sig):
    bmi_current = 0
    ak_array = np.zeros(6)
    minus_array = np.ones(6)
    ak_min = np.mean(sig[0:10])

    # 找 Ak(i)min
    for i in range(6):
        ak = np.mean(sig[10*i:10*(i+1)])
        ak_array[i] = ak
        if ak < ak_min:
            ak_min = ak
    
    # BI(k)
    inside = minus_array * ak_min
    bmi_current = np.sum(ak_array - inside)
    
    return bmi_current
    
def var_RPM(sig):
    rk_mean = np.mean(sig)
    brhr_mins = np.zeros(10)
    for i in range(10):
        brhr_mins[i] = np.mean(sig[60*i:60*(i+1)])

    return np.sum(np.square(brhr_mins - np.ones(len(brhr_mins)) * rk_mean)) / 9
    

def ada_assist(sig, brhr):

    # 0: 呼吸, 1: 心跳
    if brhr == 0:
        a = [1.5, 0.125, 0.55, 20, 5, 2, 22, 17]
    else:
        a = [1.5, 0.9, 1.9, 20, 9, 2, 5, 4]

    # Phase difference
    phase_diff = Phase_difference(sig)

    # RemoveImpulseNoise
    re_phase_diff = Remove_impulse_noise(phase_diff, int(a[0]))

    # Linear amplify
    amp_sig = Amplify_signal(re_phase_diff)

    # Bandpass signal (cheby2)
    bandpass_sig = iir_bandpass_filter_1(amp_sig, float(a[1]), float(a[2]), int(a[3]), int(a[4]), "cheby2") # Breath: 0.1 ~ 0.33 order=5, Hreat: 0.8 ~ 2.3
 
    # Smoothing signal
    smoothing_signal = MLR(bandpass_sig, int(a[5]))  # Breath = 9, Heart = 6, Delta = 1

    #detect the feature
    feature_peak, feature_valley, feature_sig = feature_detection(smoothing_signal) #找出所有的波峰及波谷

    #compress with window size 7
    compress_peak, compress_valley = feature_compress(feature_peak, feature_valley, int(a[6]), smoothing_signal)  # Br: 20 Hr: 6  ex: 25

    # Feature sort
    compress_feature = np.append(compress_peak, compress_valley)
    compress_feature = np.sort(compress_feature)

    # Candidate_search
    NT_points, _ = candidate_search(smoothing_signal, compress_feature, int(a[7]))  # breath = 18 hreat = 4 ex: 7

    return NT_points

def ada(sig, brhr):
    diff = 0
    top = ada_assist(sig, brhr)  # 0 for breath, 1 for heart.
    for top_index in range(len(top) - 1):
        cur_index = int(top[top_index + 1])
        pre_index = int(top[top_index])
        diff += math.fabs(sig[cur_index] - sig[pre_index])

    return diff

def rem_parameter(sig):
    windwos = np.zeros(5)
    for i in range(5):
        former = np.mean(sig[30*i:30*(i+1)])
        latter = np.mean(sig[30*(i+1):30*(i+2)])
        windwos[i] = math.fabs(former - latter)

    return np.mean(windwos)

def deep_parameter(bi, h):
    return bi / (h + bi)


def mov_dens_fn(raw_sig):
    count = 0
    # Segments with 0.5s length = 80 Segments
    for num in range(120):  # 80
        top = []
        first = int(num*(4))  # 0.5*20
        last = int((num+1)*(4))  # 0.5*20
        x = np.array(np.round(raw_sig[first:last], 8))
        # 方差公式
        for i in range(4):
            top.append(np.square(x[i] - np.average(x)))
        result = np.sum(top) / (4 - 1)
        if result > 0.045:  # 閥值可調
            count += 1
    percent = (count/120) * 100  # 80
    return percent

''' ----------------------- Respiration ----------------------- '''
# 10個fRSA
def tfRSA_fn(fRSA_sig):
    tfRSA = np.std(fRSA_sig)
    return tfRSA

# 31個f𝑅𝑆𝐴
def sfRSA_fn(fRSA_sig):
    sfRSA = scipy.signal.savgol_filter(fRSA_sig, 31, 3)
    sfRSA_mean = np.average(sfRSA)
    return sfRSA, sfRSA_mean

# 31個tf𝑅𝑆𝐴
def stfRSA_fn(tfRSA_sig):
    stfRSA = scipy.signal.savgol_filter(np.array(tfRSA_sig), 31, 2)
    stfRSA_mean = np.average(stfRSA)
    return stfRSA, stfRSA_mean

# 31個f𝑅𝑆𝐴
def sdfRSA_fn(fRSA, sfRSA):
    sdfRSA = np.abs(fRSA - sfRSA)
    sdfRSA = scipy.signal.savgol_filter(sdfRSA, 31, 3)
    sdfRSA_mean = np.average(sdfRSA)
    return sdfRSA, sdfRSA_mean

''' ----------------------- Heart rate ----------------------- '''
def tmHR_fn(mHR_sig):
    return tfRSA_fn(mHR_sig)

def smHR_fn(mHR_sig):
    return sfRSA_fn(mHR_sig)

def stmHR_fn(tmHR_sig):
    return stfRSA_fn(tmHR_sig)

def sdmHR_fn(mHR, smHR):
    return sdfRSA_fn(mHR, smHR)

''' ----------------------- HRV ----------------------- '''
def LF_HF_LFHF(sig):
    LF_sig = iir_bandpass_filter_1(sig, 0.04, 0.15, 20, 2, "cheby2")
    HF_sig = iir_bandpass_filter_1(sig, 0.15, 0.4, 20, 2, "cheby2")
    LF_eng = energe(LF_sig)
    HF_eng = energe(HF_sig)
    LFHF_eng = HF_eng / LF_eng
    return LF_eng, HF_eng, LFHF_eng

def energe(sig):
    N = len(sig)
    bps_fft = np.fft.fft(sig)
    return np.sum(np.square(np.abs(bps_fft[:N // 2])))

def sHF_fn(HF_sig):
    sHF = scipy.signal.savgol_filter(HF_sig, 31, 3)
    sHF_mean = np.average(sHF)
    return sHF, sHF_mean

def sLFHF_fn(LFHF_sig):
    sLFHF = scipy.signal.savgol_filter(LFHF_sig, 31, 3)
    sLFHF_mean = np.average(sLFHF)
    return sLFHF, sLFHF_mean

def time_fn(time_sig):
    time_800 = datetime.datetime.strptime('20:00:00', "%H:%M:%S")
    #print(i)
    i = datetime.datetime.strptime(str(time_sig), "%H:%M:%S")
    time_feature = i - time_800
    #print(time_feature.seconds)
    return time_feature.seconds
