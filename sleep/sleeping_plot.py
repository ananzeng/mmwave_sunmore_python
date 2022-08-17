import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.util import *
sns.set()

if __name__=="__main__":

    # Data location
    path = "./dataset_sleep/tester/"
    make_file(path)
    data_number = str(num_data(path) - 1)
    path_data = path + data_number +".csv"
    os.makedirs('./dataset_sleep/stage_fig/', exist_ok=True)

    acc_stage = np.zeros(4)
    plt.figure(figsize=(20,10))
    grid = plt.GridSpec(3, 3, wspace=0.5, hspace=0.5)
    plt.subplot(grid[0:2,:])
    saved_data = pd.read_csv(path_data)
    stage_datetime_set = []
    stage = np.array(saved_data['sleep'])
    stage_datetime = list(saved_data['datetime'])
    if len(stage_datetime) > 15:
        set_time = np.arange(0, len(stage_datetime), len(stage_datetime)//15)
        for i in set_time:
            stage_datetime_set.append(stage_datetime[i][:-3])
        plt.xticks(set_time, stage_datetime_set)

    else:
        for i in range(len(stage_datetime)):
            stage_datetime_set.append(stage_datetime[i][:-3])
        plt.xticks(np.arange(0, len(stage_datetime)), stage_datetime_set)

    # 各階段累積
    for cur_stage in stage:
        if cur_stage == 0:
            acc_stage[0] += 1
        elif cur_stage == 1:
            acc_stage[1] += 1
        elif cur_stage == 2:
            acc_stage[2] += 1
        else:
            acc_stage[3] += 1

    # 睡眠階段
    plt.plot(stage)
    plt.ylim(-1, 4)
    plt.yticks([0, 1, 2, 3], ['DEEP', 'LIGHT', 'REM', 'AWAKE'])
    plt.title('SLEEP STAGE', size=14)
    plt.ylabel('STAGE', size=14)
    plt.xlabel('TIME', size=14)

    # 各階段累積
    plt.subplot(grid[2,:])
    plt.barh(np.arange(len(acc_stage)), acc_stage, 0.4, color='royalblue')  # cornflowerblue
    plt.yticks([0, 1, 2, 3], ['DEEP', 'LIGHT', 'REM', 'AWAKE'])
    plt.xticks(np.arange(0, len(acc_stage)+1, 1))
    plt.title('SLEEP STAGE ACCUMULATION', size=14)
    plt.ylabel('STAGE', size=14)
    plt.xlabel('NUMBER OF EACH STAGE', size=14)
    plt.xticks(acc_stage, acc_stage.astype("int"))
    plt.savefig('./dataset_sleep/stage_fig/' + data_number + '.png')  # 儲存睡眠階段
    plt.show()