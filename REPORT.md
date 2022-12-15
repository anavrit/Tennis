[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# REPORT

For this project, I trained two agents in the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. The two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play. The environment was solved using the starter code provided by Udacity.

![Trained Agent][image1]

### Learning Algorithm

The Deep Deterministic Policy Gradients (DDPG) algorithm is an off-policy, model-free policy gradient model inspired from the seminal paper entitled - CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING (Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra, ICML, 2016.). To adapt it to train multiple agents, each agent receives its own, local observation. Two neural networks - one for the actor, and the other for the critic - are trained to solve the Tennis environment. The actor and critic neural networks takes in the state size input from the Tennis environment. The actor network outputs 2 continuous (action) values between -1 and 1 corresponding to movement toward (or away from) the net, and jumping. Whereas the critic network outputs 1 value corresponding to the Q value for the input state and the action taken.

**Network Architecture**

ACTOR:

`Actor(
  (fc1): Linear(in_features=8, out_features=128, bias=True)
  (bn1): BatchNorm1d(fc1)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (bn2): BatchNorm1d(fc2)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)`

The trained weights are available in `/Resources/checkpoint_actor.pth`.

CRITIC:

`Critic(
  (fcs1): Linear(in_features=8, out_features=128, bias=True)
  (bn1): BatchNorm1d(fcs1)
  (fc2): Linear(in_features=130, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)`

The trained weights are available in `/Resources/checkpoint_critic.pth`.

**Hyper-parameters**

After testing hyperparameters for batch size and learning rate, the following set of hyperparameters were used for all architectures of deep neural networks, including the selected network:

BUFFER_SIZE = int(1e5)  <br>
BATCH_SIZE = 256        <br>
GAMMA = 0.99            <br>
TAU = 1e-3              <br>
LR_ACTOR = 1e-3         <br>
LR_CRITIC = 1e-3        <br>
WEIGHT_DECAY = 0.       <br>

### Plot of Rewards

![Tennis Environment Solved](/Resources/Rewards_by_Episode.png)

The environment was solved in **1329** episodes!	Average Reward: 0.51 <br>

### Ideas for Future Work

A number of algorithms have the potential to improve the performance of DDPG for the Tennis environment. A few key ideas for future work are:

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1604.06778.pdf) (TRPO) and Truncated Natural Policy Gradient (TNPG) should achieve better performance than DDPG but could be even further improved. <br><br>
2. [D4PG](https://openreview.net/pdf?id=SyZipzbCb) or the Distributional Deterministic Deep Policy Gradient algorithm, has been shown to achieve state of the art performance on a number of challenging continuous control problems.<br><br>
3. Adding more fully-connected hidden layers and optimize hyperparameters (particularly the actor and critic learning rates) using grid search.
