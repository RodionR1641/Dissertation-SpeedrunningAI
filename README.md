# Mario Reinforcement Learning Benchmark

![Mario Gameplay Demo](media/demo.gif) 

A comprehensive benchmark of 8 deep reinforcement learning methods applied to Super Mario Bros, comparing their performance in speedrunning and completion tasks.

## 📌 Key Features
- **8 RL Algorithms Implemented**, based on 4 base algorithms(PPO,A2C,Dueling DDQN, Rainbow DQN)
- **Speedrunning Focus**: Optimized for fast level completion
- **Modular Design**: Easy to extend with new algorithms
- **WandB Integration**: Full training metrics tracking

## 📋 Contents
1. **Implemented Algorithms**
   - [x] Duelling DDQN (Double Deep Q-Network)
   - [x] PPO (Proximal Policy Optimization)
   - [x] PPO with a ViT(Vision Transformer) model  
   - [x] PPO with an LSTM layer
   - [x] PPO with RND exploration bonus (Random Network Distillation)
   - [x] A2C (Advantage Actor-Critic)
   - [x] Rainbow DQN
   - [x] Rainbow DQN with RND 

2. **Research Focus**
   - Speedrunning performance metrics of completion time and episodic reward return
   - Comparison between CNN and ViT architectures
   - Evaluation of RND as an exploration bonus

## 🛠 Requirements

### Base requirements
- Python 3.8+
- PyTorch 1.12+
- gym-super-mario-bros 7.4.0
- wandb 0.19.9+

### Full environment setup can be found in the Code/Requirements section in "user_specified.yml" conda environment setup 
conda env create -f user_specified_only.yml
pip install -r requirements.txt


## 🤖 Usage

To train the agents, the 

## 📊 Results

![Alt Text](images/diagram.png "Optional Title")

### Key Findings
1. **PPO** variations achieved fastest completion times, achieving the best time of 355 seconds on first level(completion time of 45 seconds).

2. **Implementation and hyperparameter tuning** decides the performance of an agent, regardless of a method having good theoretical foundations:
   - Our Rainbow DQN implementation was unable to consistently complete the level given the same hyperparameters as the Original Rainbow DQN paper

3. **CNN and ViT architecture matters** as overly complex models fail to capture

## 🎥 Demo GIF

This is our PPO-RND agent trained on the first level
![Mario RL Demo](Results/Videos/SuperMarioBros-1-1-v0__PPO_RND_experiment__777__2025-04-23_20-49-04-episode-5000.mp4)

##  Usage

## 📂 Repository Structure

├──Breakout/
│ ├──agent.py -> DDQN agent
│ └── ...
│
├── Mario/
│ ├── A2C -> Advantage Actor-Critic
│ ├── DQN -> Includes DDQN, Rainbow
│ ├── PPO -> All PPO variants
│ └── ...
│
├── ClusterFiles/ -> HPC/SLURM scripts
│
└── Miscellaneous/ -> Utilities and extras


## 📜 License

MIT License

Copyright (c) [2025] [Rodion Rasulov]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgements

This work borrows from various sources in Deep RL literature. We ackowledge the help such sources provided to complete this work:

Our PPO implementation follows the best practices outlined in [Huang et al. (2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

We used the OpenAI [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/) environment for our research

Other acknowledgements include:
- https://github.com/Curt-Park/rainbow-is-all-you-need/tree/master?tab=readme-ov-file
- https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/#references
- https://www.youtube.com/watch?v=OcIx_TBu90Q
- https://www.youtube.com/watch?v=_gmQZToTMac
- https://www.youtube.com/watch?v=2nonlRp3vT0&list=PLOkmXPXHDP22Lh0AAmRi7N5RlJzK68mpy