- 隨機森林權重太大[下載](https://drive.google.com/file/d/1kYTANnP9ZgzEemOCabRLeRnXyRc3I72K/view?usp=sharing "下載")至sleep/save資料夾內
- 每次重開機需要設定權限 `sudo chmod 777 /dev/ttyTHS1`

##### Create env for heart_breath
    conda create --name env_radar --file env_radar_conda.txt
    conda activate env_radar #激活並進入環境
    pip install mmwave==0.1.70
    pip install opencv-contrib-python
    pip install seaborn
    pip install keyboard
    pip install tqdm


##### Create env for sleep
    conda create --name env_radar_sleep --file env_radar_conda.txt
    conda activate env_radar_sleep #激活並進入環境
    pip install mmwave==0.1.70
    pip install opencv-contrib-python
    pip install seaborn
    pip install keyboard
    pip install tqdm
    sudo apt-get install python3-pip


##### Create env for human pose
    conda create --name env_radar_pc3 --file env_radar_pc3_conda.txt
    conda activate env_radar_pc3 #激活並進入環境
    pip install mmwave==0.1.70

- 離開環境 `conda deactivate`

| 專案內容|程式位置|conda 環境|
| ------------ | ------------ |------------ |
|呼吸心跳|heart_breath/combine_svm_people_detect_v1.0.py|env_radar|
|睡眠偵測|sleep/sleeping_v1.0.py|env_radar_sleep|
|姿勢偵測|human/human_v1.0.py|env_radar_pc3|
|動物偵測|animals/animal_v1.0.py|env_radar_pc3|

