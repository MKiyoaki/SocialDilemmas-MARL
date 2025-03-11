# Contract Augmentation of Multi-Agent Reinforcement Learning for Social Dilemmas Environments

## Final Year Project for BSc Computer Science, KCL

### Introduction

This project aims to develop contract augmentation mechanism and integrate it into EPyMARL, which is a commonly used Multi-Agent Reinforcement Learning (MARL) Algorithm Framework. 

Contract augmentation algorithms can be used to promote the capability of agents from a MARL system under social dilemmas scenarios. Typical social dilemmas including Prisoners Dilemmas, Harvest, etc. 

### Installation

- Python >= 3.11 within a virtual environment. 
  ```shell
  conda create -n sdmarl python=3.11 -y
  conda activate sdmarl
  ```
  
- PyTorch (2.5.0)
  - Check [pytorch.org](https://pytorch.org) for details.

- EPymarl
  ```shell
  cd epymarl
  pip install -r requirement.txt
  ```

- Gymnasium
  ```shell
  pip install gymnasium
  ``` 
- Shimmy
  ```shell
  pip install shimmy
  ``` 
  
- PettingZoo
  ```shell
  pip install pettingzoo
  ```
  
- MeltingPot 2.0
  ```shell
  pip install dm-meltingpot
  ```

> Alternatively, use the following command
> ```shell
> conda env create -n sdmarl -f sdmarl_requirement.yaml
> conda activate sdmarl
> ```

### Get start
