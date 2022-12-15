[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Collaboration and Competition

### Introduction

In the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

![Trained Agent][image1]

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) are added, to get a score for each agent. This yields 2 (potentially different) scores. A maximum of these 2 scores are taken.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Create and activate a new environment with Python 3.6

  **Linux or Mac:**<br>
  `conda create --name drlnd python=3.6` <br>
  `conda activate drlnd`

  **Windows:**<br>
  `conda create --name drlnd python=3.6`<br>
  `activate drlnd`    

2. Install OpenAI gym in the environment:

  `pip install gym` <br>

3. Clone the following repository and install the additional dependencies:

  `git clone https://github.com/anavrit/Deep-Q-Learning-Networks.git`<br>
  `cd Tennis`<br>
  `pip install -r requirements.txt`

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

5. Move or copy the downloaded environment to the root directory of Tennis; and unzip the file to get `Tennis.app`.

### Instructions

A brief description of files in the `Python` folder: <br>
- `agent.py`: defines a DDPG agent
- `model.py`: deep neural network model architecture
- `train.py`: code used to train the DDPG agents over multiple architectures

### Train Agent

**Note:** For MacOS users you may have to enable firewall access to Tennis.app and give access to the app through Security & Privacy settings. Instructions are [here](https://support.apple.com/guide/mac-help/block-connections-to-your-mac-with-a-firewall-mh34041/mac).

1. Navigate to the Python directory

  `cd Python`

2. Train agent with the following command:

  `python train.py`<br>

#### Resources <br>

The following key resources can be found in the `Resources` folder:

1. `checkpoint_actor.pth`: trained weights of the best actor network

2. `checkpoint_critic.pth`: trained weights of the best critic network

3. `Rewards_by_Episode.jpg`: graph tracking average reward by episode of the trained agent
