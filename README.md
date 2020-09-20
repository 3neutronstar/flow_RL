# flow-autonomous-driving

### Requirement(Installment)
- Ubuntu 18.04 is recommended. (Window OS is not supported.)
- anaconda : https://anaconda.com/
- flow-project : https://github.com/flow-project/flow
- ray-project(rllib) : https://github.com/ray-project/ray (need at least 0.8.6 is needed)
- pytorch : https://pytorch.org/

### Documentation for Flow
-English Ver: [DocumentPDF]https://drive.google.com/file/d/1NQRoCFgfIh34IJh4p0GqqOWagZh543X2/view?usp=sharing

-Korean Ver: [DocumentPDF]https://drive.google.com/file/d/1BUStOlq8LRypEmwXfRLD-_Xd04wnmCwL/view?usp=sharing

### How to Download Requirement
#### Anaconda(Python3) installation:
- Prerequisites
```shell script
    sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
- Installation(for x86 Systems)
In your browser, download the Anaconda installer for Linux (from https://anaconda.com/ ), and unzip the file. 
``` shell script
bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh
```
We recomment you to running conda init 'yes'.<br/>
After installation is done, close and open your terminal again.<br/>


#### Flow installation
Download Flow github repository.
```shell script
    git clone https://github.com/flow-project/flow.git
    cd flow
``` 
We create a conda environment and installing Flow and its dependencies within the enivronment.
```shell script
    conda env create -f environment.yml
    conda activate flow
    python setup.py develop
```
For Ubuntu 18.04: This command will install the SUMO for simulation.<br/>
```shell script
scripts/setup_sumo_ubuntu1804.sh
```
For checking the SUMO installation,
```shell script
    which sumo
    sumo --version
    sumo-gui
```
(if SUMO is installed, pop-up window of simulation is opened)
- Testing your SUMO and Flow installation
```shell script
    conda activate flow
    python simulate.py ring
```


#### Torch installation (Pytorch)
You should install at least 1.6.0 version of torch.(torchvision: 0.7.0)
```shell script
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```


#### Ray RLlib installation
You should install at least 0.8.6 version of Ray.(Recommend 0.8.7)<br/>
```shell script
pip install -U ray==0.8.7
```
- Testing RLlib installation
```shell script
    conda activate flow
    python train_rllib.py singleagent_ring
```
If RLlib is installed, turn off the terminal after confirming that "1"  appears in the part where iter is written in the terminal.


#### Visualizing with Tensorboard
To visualize the training progress:<br/>
```shell script
tensorboard --logdir=~/ray_results singleagent_ring
```

If tensorboard is not installed, you can install with pip, by following command `pip install tensorboard`.

### Downloads Flow-autonomous-driving repository 
Download related file for training and visualizing:<br/>
```shell script
cd 
git clone https://github.com/bmil-ssu/flow-autonomous-driving.git
```


## How to Use

## RL examples
### RLlib (for multiagent and single agent)

for PPO(Proximal Policy Optimization) and DDPG(Deep Deterministic Policy Gradient) algorithm
```shell script
python train_rllib.py EXP_CONFIG --algorithm [DDPG or PPO]
```

where `EXP_CONFIG` is the name of the experiment configuration file, as located in directory`exp_configs/rl/singleagent`.<br/>
In '[DDPG or PPO]', You can choose 'DDPG' or 'PPO' Algorithm.(Default: PPO)

### Visualizing Training Results
If you want to visualizing after training by rllib(ray), 
- First, ```conda activate flow``` to activate flow environment.
- Second,
```shell script
    python ~/flow-autonomous-driving/visualizer_rllib.py 
    ~/home/user/ray_results/EXP_CONFIG/experiment_name/ number_of_checkpoints
```
```experiment_name``` : Name of created folder when learning started.<br/>
```number_of_checkpoints``` : 
It means the name of the checkpoint folder created in the experiment_name folder. Enter the checkpoint (number) you want to visualize.
### Results for training Ring Network and Figure-Eight Network
#### PPO (Proximal Policy Optimization)
- Ring Network (ring length 220-270 for training)
![image](https://user-images.githubusercontent.com/59332148/91409511-78e5b780-e880-11ea-8d57-6f1d3008694a.png) <br/>
Mean velocity in 22 Non-AVs system: 4.22m/s (ring length: 260)<br/>
Mean velocity in 1 AV, 21 Non-AVs system: 4.67m/s, Mean cumulative reward: 2350 (ring length: 260) <br/>
 Use Stochastic Sampling Exploration method<br/>
 Reward seems to converge in 2300, this result is regarded as success experiment.
- Figure-eight Network
![image](https://user-images.githubusercontent.com/59332148/91409219-1ab8d480-e880-11ea-8331-7eabc58afef2.png) <br/>
Mean velocity in 14 Non-AVs system: 4.019m/s (total length: 402)<br/>
Mean velocity in 1 AV, 13 Non-AVs system: 6.67m/s (total length: 402)<br/>
 Use Gaussian Noise Exploration method<br/>
 Reward seems to converge in 19,000, this result is regarded as success experiment.<br/>
 Graph that is represented going back and forward penomenon is normal graph due to its failures.<br/>
 Having failure gives penalty to autonomous vehicle.<br/>
#### DDPG (Deep Deterministic Policy Gradient)
- Ring Network(ring length 220-270 for training)
![image](https://user-images.githubusercontent.com/59332148/91408962-b0079900-e87f-11ea-95b3-020a5809e746.png) <br/>
 Mean velocity in 22 Non-AVs system: 4.22m/s (ring length: 260)<br/>
 Mean velocity in 1 AV, 21 Non-AVs system: 4.78m/s, Mean cumulative reward: 2410 (ring length: 260) <br/>
 Use Ornstein Uhlenbeck Noise Exploration method<br/>
 
- Figure-eight Network
will be added

## non-RL examples

```shell script
python simulate.py EXP_CONFIG
```

where `EXP_CONFIG` is the name of the experiment configuration file, as located in `exp_configs/non_rl.`

If you want to run with options, use
```shell script
python simulate.py EXP_CONFIG --num_runs n --no_render --gen_emission
```
![Figure_Eight Ring](https://user-images.githubusercontent.com/59332148/91126855-f1f9d900-e6df-11ea-96ec-b3a5ee49b917.png)
    Ring Network, Figure-Eight Network(left, right)
## OSM - Output (Open Street Map)
![OSM_Combined](https://user-images.githubusercontent.com/59332148/91114406-ccaaa200-e6c2-11ea-932b-cfc2f18a6669.png)

[OpenStreetMap]https://www.openstreetmap.org/ 

If you want to use osm file for making network, Download from .osm files. After that _map.osm_ file should **replace** the same name of file in 'Network' directory.
You want to see their results, run this code.

```shell script
python simulate.py osm_test
```

After that, If you want to see those output file(XML), you could find in `~/flow/flow/core/kernel/debug/cfg/.net.cfg`



## Contributors
_BMIL at Soongsil Univ._
Prof. Kwon (Minhae Kwon), 
Minsoo Kang, 
Gihong Lee, 
Hyeonju Lim
