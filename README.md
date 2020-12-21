# muzero - pytorch implementation plays cartpole

pytorch implementation of ["Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"](https://arxiv.org/abs/1911.08265) based on his pseudocode. This implementation is intended to be as close as possible to the pseudocode presented.

## How is this implementation different with respect to the original paper? 
The main difference is that this version uses the uniform distribution to samples data from the replay, instead of using prioratized experience replay.<br>

## Muzero plays cartpole

To train your own muzero to play with caterpole you just have to launch muzero_main.py. <br>
To evaluate the average sum of rewards it gets (number of moves that performs before failing (or finishing) the game in the case of caterpole), you can call the test.py function.<br>

Some metrics that it's possible to keep track while training (using tensorboard):

mean_reward: mean rewards of the last 50 games <br>
<img src="https://github.com/Atze00/muzero/blob/main/images/mean_reward.png" width="480">

policy_loss:<br>
<img src="https://github.com/Atze00/muzero/blob/main/images/policy_loss.png" width="480">

value_loss:<br>
<img src="https://github.com/Atze00/muzero/blob/main/images/value_loss.png" width="480">

reward_loss:<br>
<img src="https://github.com/Atze00/muzero/blob/main/images/reward_loss.png" width="480">

total_loss:<br>
<img src="https://github.com/Atze00/muzero/blob/main/images/total_loss.png" width="480">

## What scores can I expect to get with caterpole?

Getting a score of 200-250+ is very feasable without tweaking parameters. <br>
The problem with cartpole is that the training replay gets less and less crowded with failed games, using prioritized experience replay can be a solution to this problem.<br>
