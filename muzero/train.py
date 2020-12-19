import ray
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import torch.optim as optim
from muzero.utils import SharedStorage,ReplayBuffer,Transform
from muzero.selfplay import run_selfplay
from time import sleep

def muzero(config):
    storage = SharedStorage.remote(config)
    replay_buffer = ReplayBuffer.remote(config)
    writer = SummaryWriter() 
    for _ in range(config.num_actors):
        run_selfplay.remote(config, storage, replay_buffer)
    train_network(config, storage, replay_buffer,writer)
    return ray.get(storage.latest_network.remote())

def train_network(config, storage, replay_buffer, writer):
    network = config.make_uniform_network()
    learning_rate = config.lr_init
    optimizer = optim.SGD(network.parameters(),learning_rate, config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10000],gamma = 0.5)
    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network.remote(i, network)
        while ray.get(replay_buffer.get_len.remote()) <= 0:
            sleep(1)
        batch = ray.get(replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps))
        update_weights(optimizer, network.train(), batch, config.weight_decay,writer, i,config.num_unroll_steps,config.range_v)
        writer.add_scalar("mean_reward",ray.get(replay_buffer.get_mean_rewards.remote()),i)
        scheduler.step()
    storage.save_network.remote(config.training_steps, network.eval().cpu())


def update_weights(optimizer, network, batch, weight_decay, writer, iteration_n, num_unroll_steps, range_v):
    loss = 0
    l_v = 0 
    l_p = 0
    l_r = 0
    optimizer.zero_grad()
    binary_cross_entropy = torch.nn.BCELoss()
    images, actions, targets = batch
    target_values, target_rewards, target_policy = targets 
    target_policy =  target_policy
    target_values = Transform.transform_values(target_values,range_v)
    target_rewards = Transform.transform_values(target_rewards,range_v)
    gradient_scale = 1
    values, rewards, policy_logits, hidden_states = network.initial_inference(images)
    for i in range(num_unroll_steps+1):
        if i > 0:
            gradient_scale = 1/(num_unroll_steps)
            values, rewards, policy_logits, hidden_states = network.recurrent_inference(
                    hidden_states, actions[:,i-1].unsqueeze(-1))
            hidden_states.register_hook(lambda grad: grad*0.5)

        l_v += binary_cross_entropy(F.softmax(values,dim=1), target_values[:,i])*gradient_scale
        l_p += binary_cross_entropy(F.softmax(policy_logits,dim=1), target_policy[:,i])*gradient_scale
        l_r +=binary_cross_entropy(F.softmax(rewards,dim=1), target_rewards[:,i])*gradient_scale 
    loss = l_v+l_p+l_r
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(),1)
    optimizer.step()    
    writer.add_scalar('value_loss', l_v.item(), iteration_n)
    writer.add_scalar('reward_loss', l_r.item(), iteration_n)
    writer.add_scalar('policy_loss', l_p.item(), iteration_n)
    writer.add_scalar('tot_loss', loss.item(), iteration_n)


    


