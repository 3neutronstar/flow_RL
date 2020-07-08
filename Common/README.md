# flow-autonomous-driving

### Requirement

- flow-project : https://github.com/flow-project/flow
- anaconda : https://anaconda.com/

## How to Use

'''shell script
python simulate.py "test"
'''
This examples is for test the simulate.py for non-RL problem.

## non-RL examples

```shell script
python simulate.py EXP_CONFIG
```

where `EXP_CONFIG` is the name of the experiment configuration file, as located in `exp_configs/non_rl.`

```shell script
 python simulate.py EXP_CONFIG --num_runs n --no_render --gen_emission
```

## RL examples

### RLlib

```shell script
python train.py EXP_CONFIG --rl_trainer "rllib"
```

where `EXP_CONFIG` is the name of the experiment configuration file, as located in `exp_configs/rl/singleagent` or `exp_configs/rl/multiagent.`

### stable-baselines

traffic light agents being trained through RL algorithms provided by OpenAI _stable-baselines_.

```shell script
python simulate.py EXP_CONFIG --rl_trainer "stable-baselines"
```
