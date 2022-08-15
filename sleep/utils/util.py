import os
import csv
import numpy as np
import datetime

def make_file(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def num_data(path):
    num = len(os.listdir(path)) // 2
    return num

def define_time(next_YMD, start_year, start_month, start_day, start_time):
    ct2 = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    end_year = ct2[0:4]  
    end_month = ct2[5:7]
    end_day = ct2[8:10]
    end_time = int(ct2[11:13])*3600 + int(ct2[14:16])*60 + int(ct2[17:19])

    # 當隔天為: 過年、過月、過天
    if int(end_year) - int(start_year) >= 1:
        # 換起始時間
        start_year = end_year
        start_month = end_month
        start_day = end_day
        start_time = end_time
        next_YMD = True
    elif int(end_month) - int(start_month) >= 1:
        # 換起始時間
        start_month = end_month
        start_day = end_day
        start_time = end_time
        next_YMD = True
    elif int(end_day) - int(start_day) >= 1:
        # 換起始時間
        start_day = end_day
        start_time = end_time
        next_YMD = True
    return next_YMD, start_year, start_month, start_day, start_time, end_time, ct2[17:19]

def recording(path_data, vd, hr_rpm, br_rpm):
    ct3 = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") # 時間格式為字串
    with open(path_data, "a",newline="") as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow([vd.rangeBinIndexMax,vd.rangeBinIndexPhase,vd.maxVal,vd.processingCyclesOut,vd.processingCyclesOut1,
                        vd.rangeBinStartIndex,vd.rangeBinEndIndex,vd.unwrapPhasePeak_mm,vd.outputFilterBreathOut,vd.outputFilterHeartOut,
                        vd.heartRateEst_FFT,vd.heartRateEst_FFT_4Hz,vd.heartRateEst_xCorr,vd.heartRateEst_peakCount,vd.breathingRateEst_FFT,
                        vd.breathingEst_xCorr,vd.breathingEst_peakCount,vd.confidenceMetricBreathOut,vd.confidenceMetricBreathOut_xCorr,vd.confidenceMetricHeartOut,
                        vd.confidenceMetricHeartOut_4Hz,vd.confidenceMetricHeartOut_xCorr,vd.sumEnergyBreathWfm,vd.sumEnergyHeartWfm,vd.motionDetectedFlag,
                        vd.rsv[0],vd.rsv[1],vd.rsv[2],vd.rsv[3],vd.rsv[4],vd.rsv[5],vd.rsv[6],vd.rsv[7],vd.rsv[8],vd.rsv[9],ct3[11:19], hr_rpm, br_rpm])

def recording_final(path_data, current_time, all_results, sleep):
    ct3 = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") # 時間格式為字串
    with open(path_data, "a", newline="") as csvFile:
        writer = csv.writer(csvFile, dialect = "excel")
        writer.writerow([all_results[22], all_results[23], all_results[0], all_results[1], all_results[2], all_results[3], all_results[4], all_results[5], all_results[6], all_results[7], all_results[8], all_results[9], all_results[10],
                        all_results[11], all_results[12], all_results[13], all_results[14], all_results[15], all_results[16], all_results[17], all_results[18], all_results[19], all_results[20], all_results[21],
                        current_time, sleep])