- 隨機森林權重太大[下載](https://drive.google.com/file/d/1kYTANnP9ZgzEemOCabRLeRnXyRc3I72K/view?usp=sharing "下載")至sleep/save資料夾內
- 每次重開機需要設定權限 `sudo chmod 777 /dev/ttyTHS1`

##### Create env for heart_breath
    conda create --name env_radar --file env_radar_conda.txt
    conda activate env_radar
    pip install mmwave==0.1.70
    pip install opencv-contrib-python
    pip install seaborn
    pip install keyboard
    pip install tqdm


##### Create env for sleep
    conda create --name env_radar_sleep --file env_radar_conda.txt
    conda activate env_radar_sleep
    pip install mmwave==0.1.70
    pip install opencv-contrib-python
    pip install seaborn
    pip install keyboard
    pip install tqdm
    sudo apt-get install python3-pip
    python sleeping.py


##### Create env for human pose
    conda create --name env_radar_pc3 --file env_radar_pc3_conda.txt
    conda activate env_radar_pc3
    pip install mmwave==0.1.70

| 專案內容|程式位置|
| ------------ | ------------ |
|呼吸心跳|heart_breath/combine_svm.py|
|睡眠偵測|sleep/sleeping.py|
|姿勢偵測|human_2022_07_13/human_2022_07_13.py|
|動物偵測|animals/animal.py|

