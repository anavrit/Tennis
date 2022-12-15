from unityagents import UnityEnvironment
import numpy as np

import random
import copy
import matplotlib.pyplot as plt

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import Agents

env = UnityEnvironment(file_name="../Tennis.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

agents = Agents(state_size=state_size, action_size=action_size, num_agents=len(env_info.agents), random_seed=5)

def ddpg(n_episodes=2000, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agents.reset()
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agents.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations             # get the next states
            rewards = env_info.rewards                             # get the rewards
            dones = env_info.local_done                            # see if the episode has finished for any agent
            agents.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break

        scores_deque.append(np.max(score))
        scores.append(np.max(score))

        if i_episode % print_every == 0:
            print('\rEpisode {}... Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes... Average Score: {:.2f}'.format(i_episode - 100, np.mean(scores_deque)))
            torch.save(agents.actor_local.state_dict(), '../Resources/checkpoint_actor.pth')
            torch.save(agents.critic_local.state_dict(), '../Resources/checkpoint_critic.pth')
            break

    return scores

scores = ddpg()

torch.save(agents.actor_local.state_dict(), '../Resources/checkpoint_actor.pth')
torch.save(agents.critic_local.state_dict(), '../Resources/checkpoint_critic.pth')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.title('Reward vs. Episode')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.grid(True)
plt.show()

env.close()
