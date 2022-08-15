import numpy as np
import pandas as pd

import os
import scipy

def break_off_both_ends(path, sig):
    for name in os.listdir(path):
        print("正在處理：", name)
        df = pd.read_csv(os.path.join("dataset_sleep_test", name))

        for i in range(df.shape[0]-1500, df.shape[0], 1):
            sec = df['datetime'][i][-2:]
            if  sec == "00":
                break

        for i in range(1500):
            sec = df['datetime'][i][-2:]
            if  sec == "00":
                break
    df.drop(df.index[0:i],inplace=True)  # 刪除1,2行的整行數據
    df.to_csv(os.path.join("dataset_sleep_test", "processed_data", name[:-4]+ "_processed_data.csv"),index=False,encoding="utf-8")