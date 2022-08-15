import numpy as np
import time
from scipy import signal
from scipy.signal import find_peaks
from scipy.fftpack import fft

def Phase_difference(unwarp_phase):
	phase_diff = []
	for tmp in range(len(unwarp_phase)):
		if tmp > 0:
			phase_diff_tmp = unwarp_phase[tmp] - unwarp_phase[tmp - 1]
			phase_diff.append(phase_diff_tmp)
	return phase_diff

def Remove_impulse_noise(phase_diff, thr):
	removed_noise = np.copy(phase_diff)
	for i in range(1, len(phase_diff)-1):
		forward = phase_diff[i] - phase_diff[i-1]
		backward = phase_diff[i] - phase_diff[i+1]
		#print(forward, backward)
		if (forward > thr and backward > thr) or (forward < -thr and backward < -thr):
			removed_noise[i] = phase_diff[i-1] + (phase_diff[i+1] -  phase_diff[i-1])/2
		removed_noise[i] = phase_diff[i]
	return removed_noise

def Amplify_signal(removed_noise):
	for i in range(len(removed_noise)):
		tmp = removed_noise[i]
		if tmp > 0:
			tmp += 1
		elif tmp < 0:
			tmp -= 1
		tmp *= 5
		removed_noise[i] == tmp
	return removed_noise
	
def iir_bandpass_filter_1(data, lowcut, highcut, signal_freq, filter_order, ftype):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = signal.iirfilter(filter_order, [low, high], rp=5, rs=60, btype="bandpass", ftype = ftype)
    y = signal.lfilter(b, a, data)
    return y

def MLR(data, delta):
    data_s = np.copy(data)
    mean = np.copy(data)
    m = np.copy(data)
    b = np.copy(data)
    for t in range(len(data)):
        if (t - delta ) < 0 or (t + delta + 1) > len(data):
            None
        else:
            start = t - delta
            end = t + delta + 1
            mean[t] = np.mean(data[start:end])

            mtmp = 0
            for i in range(-delta, delta + 1):
                mtmp += i * (data[t + i] - mean[t])
            m[t] = (3 * mtmp) / (delta * (2 * delta + 1) * (delta + 1))
            b[t] = mean[t] - (t * m[t])

    for t in range(len(data)):
        if (t - delta) < 0 or (t + delta + 1) > len(data):
            data_s[t] = data[t]
        else:
            tmp = 0
            for i in range(t - delta, t + delta):
                tmp += m[i] * t + b[i]
            data_s[t] = tmp / (2 * delta + 1)
    return data_s

def feature_detection(data):
	data_v=np.copy(data)
	feature_peak, _ = find_peaks(data)
	feature_valley, _  = find_peaks(-data)
	data_v=np.multiply(np.square(data),np.sign(data))
	return feature_peak,feature_valley,data_v

def feature_compress(feature_peak,feature_valley,time_thr,signal):
	feature_compress_peak=np.empty([1,0])
	feature_compress_valley=np.empty([1,0])
	# sort all the feature
	feature=np.append(feature_peak,feature_valley)

	feature=np.sort(feature)
	# grouping the feature
	ltera=0
	while(ltera < (len(feature)-1)):
		# record start at valley or peak (peak:0 valley:1)
		i, = np.where(feature_peak == feature[ltera])
		if(i.size==0):
			start=1
		else:
			start=0
		ltera_add=ltera
		while(feature[ltera_add+1]-feature[ltera_add]<time_thr):
			# skip the feature which is too close
			ltera_add=ltera_add+1
			#break the loop if it is out of boundary
			if(ltera_add >= (len(feature)-1)):
				break
		# record end at valley or peak (peak:0 valley:1)
		i, = np.where(feature_peak == feature[ltera_add])
		if(i.size==0):
			end=1
		else:
			end=0
		# if it is too close
		if (ltera!=ltera_add):
			# situation1: began with valley end with valley
			if(start==1 and end==1):
				# using the lowest feature as represent
				tmp=(np.min(signal[feature[ltera:ltera_add]]))
				i, = np.where(signal[feature[ltera:ltera_add]] == tmp)
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera+i])
			#situation2: began with valley end with peak
			elif(start==1 and end==0):
				# using the left feature as valley, right feature as peak
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera])
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera_add])
			#situation3: began with peak end with valley
			elif(start==0 and end==1):
				# using the left feature as peak, right feature as valley
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera])
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera_add])
			#situation4: began with peak end with peak
			elif(start==0 and end==0):
				# using the highest feature as represent
				# tmp=np.array(tmp,dtype = "float")
				tmp= np.max(signal[feature[ltera:ltera_add]])
				i, = np.where(signal[feature[ltera:ltera_add]] == tmp)
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera+i])
			ltera=ltera_add
		else:
			# it is normal featur point
			if(start):
				feature_compress_valley=np.append(feature_compress_valley,feature[ltera])
			else:
				feature_compress_peak=np.append(feature_compress_peak,feature[ltera])				
		ltera=ltera+1

	return feature_compress_peak,feature_compress_valley

def candidate_search(signal_v,feature,window_size):
	NT_point=np.empty([1,0])
	NB_point=np.empty([1,0])
	#doing the zero paddding
	signal_pad=np.ones((len(signal_v)+2*window_size))
	signal_pad[window_size:(len(signal_pad)-window_size)]=signal_v
	signal_pad[0:window_size]=signal_v[0]
	signal_pad[(len(signal_pad)-window_size):-1]=signal_v[-1]
	# calaulate the mean and std using windows(for peaks)
	for i in range(len(feature)):
		# for the mean
		window_sum=(np.sum(signal_pad[int(feature[i]):int(feature[i]+2*window_size+1)]))/(window_size*2+1)
		window_var=np.sqrt(np.sum(np.square(signal_pad[int(feature[i]):int(feature[i]+2*window_size+1)]-window_sum))/(window_size*2+1))
		#determine if it is NT
		if(signal_v[feature[i].astype(int)]>window_sum and window_var>0.01):
			NT_point=np.append(NT_point,feature[i])
		# determine if it is BT
		elif(signal_v[feature[i].astype(int)]<window_sum and window_var>0.01):
			NB_point=np.append(NB_point,feature[i])

	return NT_point,NB_point

def caculate_breathrate(NT_points,NB_points):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# if both NT and NB are not detected
    if NT_points.shape[0] <= 1 and NB_points.shape[0] <= 1:
        return 0
    # if only NT are detected
    elif NT_points.shape[0] > 1 and NB_points.shape[0] <= 1:
        tmp = np.concatenate(([0], NT_points), axis=0)
        tmp_2 = np.concatenate((NT_points, [0]), axis=0)
        aver_NT = tmp_2[1:-1] - tmp[1:-1]
        return 1200 / np.mean(aver_NT)  # (60)*(20)
    # if only NB are detected
    elif NB_points.shape[0] > 1 >= NT_points.shape[0]:
        tmp = np.concatenate(([0], NB_points), axis=0)
        tmp_2 = np.concatenate((NB_points, [0]), axis=0)
        aver_NB = tmp_2[1:-1] - tmp[1:-1]
        return 1200 / np.mean(aver_NB)
    else:
        tmp = np.concatenate(([0], NT_points), axis=0)  # tmp 兩點距離
        tmp_2 = np.concatenate((NT_points, [0]), axis=0)
        aver_NT = tmp_2[1:-1] - tmp[1:-1]
        tmp = np.concatenate(([0], NB_points), axis=0)
        tmp_2 = np.concatenate((NB_points, [0]), axis=0)
        aver_NB = tmp_2[1:-1] - tmp[1:-1]
        aver = (np.mean(aver_NB) + np.mean(aver_NT)) / 2
    return 1200 / aver    #因為一個周期是20Hz所以時間就是1/20，然後是算每分鐘，所以還要再除以1/60

def detect_Breath(unw_phase, a): #,lowHz
	replace = False
	N = 0
	T = 0
	# Phase difference
	phase_diff = Phase_difference(unw_phase)

	# RemoveImpulseNoise
	re_phase_diff = Remove_impulse_noise(phase_diff, int(a[0]))

	# Linear amplify
	amp_sig = Amplify_signal(re_phase_diff)

	# Bandpass signal (cheby2)
	bandpass_sig = iir_bandpass_filter_1(amp_sig, float(a[1]), float(a[2]), int(a[3]), int(a[4]), "cheby2") # Breath: 0.1 ~ 0.33 order=5, Hreat: 0.8 ~ 2.3
	N = len(bandpass_sig)
	T = 1 / 20
	bps_fft = fft(bandpass_sig)
	bps_fft_x = np.linspace(0, 1.0 / (T * 2), N // 2)
	index_of_fftmax = np.argmax(2 / N * np.abs(bps_fft[:N // 2])) * (1.0 / (T * 2)) / (N // 2)

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
	NT_points, NB_points = candidate_search(smoothing_signal, compress_feature, int(a[7]))  # breath = 18 hreat = 4 ex: 7

	rate = caculate_breathrate(NT_points, NB_points)
	return rate, index_of_fftmax

def timing(sec):
	print(f"Get ready in {sec} seconds...")
	current_time = 0
	print(f"{current_time} sec")
	while current_time != sec:
		time.sleep(1)
		current_time += 1
		print(f"{current_time} sec")

"""
當呼吸律或心律為異常值, 以前一秒的輸出值取代。
"""
def substitute(pre, input, result_type):
	if result_type == 0:
		if input < 40 or input > 110:
			return pre
		else:
			return input
	else:
		if input < 10 or input > 25:
			return pre
		else:
			return input