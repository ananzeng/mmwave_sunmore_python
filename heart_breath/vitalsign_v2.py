# vitalsign
# ver:0.0.10
# 2020/02/04

import serial
import struct

# Vital Signs Output Stats (vsos) (生理訊號輸出資料)
class Vsos:
    rangeBinIndexMax = 0
    rangeBinIndexPhase = 0
    maxVal = float(0.0)
    processingCyclesOut = 0
    processingCyclesOut1 = 0
    rangeBinStartIndex = 0
    rangeBinEndIndex = 0
    unwrapPhasePeak_mm = float(0.0)
    outputFilterBreathOut = float(0.0)
    outputFilterHeartOut = float(0.0)
    heartRateEst_FFT = float(0.0)
    heartRateEst_FFT_4Hz = float(0.0)
    heartRateEst_xCorr = float(0.0)
    heartRateEst_peakCount = float(0.0)
    breathingRateEst_FFT = float(0.0)
    breathingEst_xCorr = float(0.0)
    breathingEst_peakCount = float(0.0)
    confidenceMetricBreathOut = float(0.0)
    confidenceMetricBreathOut_xCorr = float(0.0)
    confidenceMetricHeartOut = float(0.0)
    confidenceMetricHeartOut_4Hz = float(0.0)
    confidenceMetricHeartOut_xCorr = float(0.0)
    sumEnergyBreathWfm = float(0.0)
    sumEnergyHeartWfm = float(0.0)
    motionDetectedFlag = float(0.0)
    rsv = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class Header:
    magicWord = [b'\x02', b'\x01', b'\x04', b'\x03', b'\x06', b'\x05', b'\x08', b'\x07']  # , b'\0x99' 沒差?
    version = ''
    totalPackLen = 0
    tvlHeaderLen = 8
    platform = ''
    frameNumber = 0
    timeCpuCycles = 0
    numDetectedObj = 0
    numTLVs = 0
    rsv = 0


class VitalSign:
    port = ''
    header = Header()
    vs = Vsos()

    def __init__(self, port):
        self.port = port
        print(f'Vital sign init: {self.port}')

    def vital_port(self):
        print(f'Vital sign connnected: {self.port}')

    # 可以看雷達輸入資料的整體長度，配合 serial.serial 的 timeout
    def showHeader(self):
        print('Show Header...')
        print(f'Frame Number:\t {self.header.frameNumber}')
        print(f'Version:\t {self.header.version}')
        print(f'TLV:\t\t {self.header.numTLVs}')
        print(f'Number of Detected Obj: {self.header.numDetectedObj}')
        print(f'Platform:\t {self.header.platform}')
        print(f'Total Pack Len:\t {self.header.totalPackLen}')

    def getvitalSignsOutputStats(self):
        return self.vs

    def getHeader(self):
        return self.header

    def vitalSignsOutputStats(self, buf):
        try:
            (self.vs.rangeBinIndexMax,
             self.vs.rangeBinIndexPhase,
             self.vs.maxVal,
             self.vs.processingCyclesOut,
             self.vs.processingCyclesOut1,
             self.vs.rangeBinStartIndex,
             self.vs.rangeBinEndIndex,
             self.vs.unwrapPhasePeak_mm,
             self.vs.outputFilterBreathOut,
             self.vs.outputFilterHeartOut,
             self.vs.heartRateEst_FFT,
             self.vs.heartRateEst_FFT_4Hz,
             self.vs.heartRateEst_xCorr,
             self.vs.heartRateEst_peakCount,
             self.vs.breathingRateEst_FFT,
             self.vs.breathingEst_xCorr,
             self.vs.breathingEst_peakCount,
             self.vs.confidenceMetricBreathOut,
             self.vs.confidenceMetricBreathOut_xCorr,
             self.vs.confidenceMetricHeartOut,
             self.vs.confidenceMetricHeartOut_4Hz,
             self.vs.confidenceMetricHeartOut_xCorr,
             self.vs.sumEnergyBreathWfm,
             self.vs.sumEnergyHeartWfm,
             self.vs.motionDetectedFlag,
             self.vs.rsv[0], self.vs.rsv[1], self.vs.rsv[2],
             self.vs.rsv[3], self.vs.rsv[4], self.vs.rsv[5],
             self.vs.rsv[6], self.vs.rsv[7], self.vs.rsv[8],
             self.vs.rsv[9]) = struct.unpack('2Hf2H2H28f', buf)
        except:
            print('Improper VSOS!')
            return False, self.vs

        return True, self.vs

    def tlv_read(self, disp):
        vsdata = Vsos()
        sbuf = b''
        data_idx = 0
        structure_idx = 'magicWord'

        # Message TLV Header 1 (8 Bytes)
        mmWaveType1 = 0
        length1 = 0
        # Message TLV Header 2 (8 Bytes)
        mmWaveType2 = 0
        length2 = 0

        while True:
            reader = self.port.read()
            # print(reader)
            # --------------------------------- magicWord ---------------------------------
            if structure_idx == 'magicWord':
                if reader == self.header.magicWord[data_idx]:
                    data_idx += 1
                    if data_idx == 8:
                        data_idx = 0
                        structure_idx = 'header'
                        rangeProfile = b''
                        sbuf = b''
                else:
                    data_idx = 0
                    rangeProfile = b''
                    return False, vsdata, rangeProfile

            # -------- Header without magicWord(32 Bytes) + Message TLV Header(8 Bytes) = 40 Bytes --------
            elif structure_idx == 'header':
                sbuf += reader
                data_idx += 1
                if data_idx == 40:
                    data_idx = 0
                    
                    try:
                        (self.header.version, self.header.totalPackLen, self.header.platform,
                         self.header.frameNumber, self.header.timeCpuCycles, self.header.numDetectedObj,
                         self.header.numTLVs, self.header.rsv, mmWaveType1, length1) = struct.unpack('10I', sbuf)

                    except:
                        print('Improper Header!')
                        return False, vsdata, rangeProfile

                    if disp == True:
                        self.showHeader()
                        print(f'VSOS Type: {mmWaveType1}')
                        print(f'VSOS Len: {length1}')

                    structure_idx = 'VSOS'
                    sbuf = b''

                elif data_idx > 40:
                    data_idx = 0
                    structure_idx = 'magicWord'
                    return False, vsdata, rangeProfile

            # -------- Message TLV Header(8 Bytes) --------
            # elif structure_idx == 'Message_TLV_Header_1':
            #     sbuf += reader
            #     data_idx += 1
            #     if data_idx == 8:
            #         try:
            #             (mmWaveType1, length1) = struct.unpack('2I', sbuf)
            #             print(mmWaveType1, length1)
            #         except:
            #             print('Improper Message_TLV_Header_1!')
            #             return False, vsdata, rangeProfile

            #         if disp == True:
            #             self.showHeader()
            #             print(f'VSOS Type: {mmWaveType1}')
            #             print(f'VSOS Len: {length1}')

            #         data_idx = 0
            #         structure_idx = 'VSOS'
            #         sbuf = b''
            #         length1 = 128  # Vital Signals Output Stats = 128 Bytes

            #     elif data_idx > 8:
            #         data_idx = 0
            #         structure_idx = 'magicWord'
            #         return False, vsdata, rangeProfile

            # -------------------------- Vital Signs Output Status = 128 Bytes --------------------------
            elif structure_idx == 'VSOS':
                sbuf += reader
                data_idx += 1
                if  data_idx == length1:
                    data_idx = 0

                    try:
                        vflag, vsdata = self.vitalSignsOutputStats(sbuf)

                    except:
                        print('Improper vitalSignsOutputStats!')

                    if not vflag:
                        structure_idx = 'magicWord'
                        return False, vsdata, rangeProfile

                    structure_idx = 'Message_TLV_Header_2'
                    sbuf = b''

                elif data_idx > length1:
                    data_idx = 0
                    structure_idx = 'magicWord'
                    return False, vsdata, rangeProfile

            # -------------------------- Message TLV Header 2 = 8 Bytes --------------------------
            elif structure_idx == 'Message_TLV_Header_2':
                sbuf += reader
                data_idx += 1
                if data_idx == 8:
                    data_idx = 0

                    try:
                        mmWaveType2, length2 = struct.unpack('2I', sbuf)

                    except:
                        print('Improper Message_TLV_Header_2!')
                        return False, vsdata, rangeProfile

                    if disp == True:
                        print(f'Range Profile Type: {mmWaveType2}')
                        print(f'Range Profile Len: {length2}')

                    if length2 > 252:
                        length2 = 252  # 資料的總長
                    structure_idx = 'rangeProfile'
                    rangeProfile = b''

                elif data_idx > 8:
                    data_idx = 0
                    structure_idx = 'magicWord'

            # -------------------------- Range profile = length2/2 Bytes --------------------------
            elif structure_idx == 'rangeProfile':
                rangeProfile += reader
                data_idx += 1
                if data_idx == length2:
                    data_idx = 0

                    try:
                        fmt = f'{int(length2 / 2)}h'  # {:d} 整數的意思
                        xd = struct.unpack(fmt, rangeProfile)

                    except:
                        print('Improper rangeProfile!')
                        return False, vsdata, rangeProfile

                    if disp == True:
                        print("---------rangeProfile:" + str(len(rangeProfile)))
                        print(":".join("{:02x}".format(c) for c in rangeProfile))
                    return True, vsdata, list(xd)  # True: 確認讀到資料 > vsdata: Vitial Signs data > list(xd): Range profile
                elif data_idx > length2:
                    data_idx = 0
                    structure_idx = 'magicWord'
