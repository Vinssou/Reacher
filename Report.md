# Project 2: Continuous Control

## Introduction

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


## Evolution Strategy

I have been fascinated by one policy based approach which is the evolution strategy, simply because it uses the same strategy of evolution, and so randomness is the main tool. I wanted to solve this problem using this method although I knew it was not the most efficient method for this problem. First I solved quite easily the [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) from open Ai gym, just by adapting a few hyperparameters. 
To solve the Reacher environment, I managed to reach the score 3 with the first approach so I tried to improve the algorithm using various techniques.

### Architecture
When I could solve the [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) walker without a hidden layer, this seemed completely impossible for the complexity of this problem. I added a hidden layer of 128 neurons to solve this problem. This already shows a good improvement.

### Adaptive Noise
I introduced in the algorithm an adaptive scaler that increases the standard deviation when the algorithm is not progressing, and decreases it otherwise. This is allowing exploration in an adaptive way. 

### Drop out layers
In the evolution algorithm there is no backward propagation. I wanted to find a way not to change all neurons at the same time, and make evolve a subset of those neurons. When generating a new noise to update the new weight I wanted to disable some neurons, in the same way we use dropout layers. To do that I used a filer that was disabling the new generating noise for 20% of the neurons : 
```python
new_weight = previous_best_generation_weight + np.random.choice(2, size=weight_count, p=[0.2, 0.8]) * noise_for_the_next_generation
```
Similarly than the dropout layer, 20% of the neurons were fixed during this evolution. This technique brought a significant improvement. 

### Weighted average generation 
In the original evolution the new generation is based on the percentage of the population. For example we keep 20% of the best new population. The mean of the network weights of this percentage is used to create the new generation.
Here instead of using the mean, I used a weighted average, so to build the next generation the strongest agent will use more of its weight than the second best agent and so on. I used this technique inspired by evolution in life. The most powerful in a herd usually reproduce more. 

```python
avg_weight = np.arange(elite_population_count) + np.ones(elite_population_count)
# elite_weights : Elite weights selected for the next generation sorted from weaker to stronger
best_weight_avg = np.average(elite_weights, axis=0,  weights=avg_weight)
```

### Various generation
For the next generation I used three types of new generation: 
- One is generation is using the mean of the weights
- The other is using a weighted average, 1 for the weaker, 2 for the one stronger of the weaker...etc and so on and n for the strongest one.
- Another generation uses the score as weight for the weighted average. I took advantage that the rewards are positive in this environment to compute a specific weighted average. 
```python
best_weight_rewards = np.average(elite_weights, axis=0,  weights=elite_rewards)
```

### Conclusion
The evolution strategy is impressive to solve really complex problems without using backpropagation. This technique is based only on generating random weights. By the previous improvement described I could increase the score from 3 to reach above 14.
All the source code with the various improvement is available [here](https://github.com/Vinssou/ReacherEvolution/blob/master/Continuous_Control.ipynb)

![Progress](evolution_progresss01.png)

## DDPG
Although the Evolution Strategy shows impressive results, I couldn't solve this environment in a decent amount of time. My second attempt was using the DDPG architecture. 

### Network Architecture
This architecture used two neural networks, an actor that predict the action, and a critic that predict the value function. Both of those neural networks used a target network. So there are 4 neural networks in total.
The actor and the critic are composed of one hidden layer of 128 neurons, and I used a learning rate of 0.001.

### DDPG Learning
The critic network takes as input the state and the action of a certain step, and predicts how good is the situation according to the state and action. Then the actor tries to predict the best action accordingly to the critic. The policy gradient method has high variance. Using both actor and critic brings accuracy and stability to the learning.  
In addition to the Actor Critic architecture DDPG uses target networks for both actor and critic, that solve the overestimate problem.

### Soft update
At every step we update the weight of the actor and the critic target networks, by a small amount. 

### Batch Learning
At first no learning seemed to happen; the score was stuck below 1.0. Then I started to process the learning more frequently. Each step I was calling the learn function several time. I was also trying to increase or decrease the learning batch from 32 to 64, but the algorithm was not improving much. Then I stopped to process the learning at every step and then the score started to increase. At glance, what surprised me was that by increasing the batch number even up to 256, the learning was much faster. 

### Hyperparameters

```python
BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

EXPLORATION_DECAY = 0.999999
EXPLORATION_MIN = 0.01
LEARN_FREQUENCY = 20*20
LEARN_COUNT = 10
```

### Resolution
This environment has been solved in 120 episodes.
![Progress](ddpg_solved.png)

### Conclusion
The DDPG algorithm seemed not to work at the beginning I had much better result with the Evolution Strategy. But by learning by batch with DDPG, it changed drastically the learning. I was really surprised by the impact of such a change. I could solve this environment within 120 episodes.

## Future Work
I am planning to implement an Actor Critic method, using Proximal Policy Optimization for the actor, I am planning to use the Generalized Advantage Estimate. The critic would be a DQN or a D3QN. With this method I would be able to solve the environment in less episod. 

