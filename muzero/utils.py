import ray
import numpy as np
import torch.nn.functional as F
import torch 
import gym
import collections

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class Transform():
    @staticmethod
    def h_transform(x):
        e = 10**-2
        sign = torch.ones((x.shape)).to(x.device)
        sign[x<0] = -1
        return sign*(torch.sqrt(torch.abs(x)+1)-1)+ e*x

    @staticmethod
    def inverse_h_transform(x):
        e = 10**-2
        sign = torch.ones(x.shape).to(x.device)
        sign[x<0] = -1
        return sign*(((torch.sqrt(1+4*e*(torch.abs(x)+1+e))-1)/(2*e))**2-1) 

    @staticmethod
    def phi_t(x,min_value,max_value):
        y = torch.zeros((x.shape[0],x.shape[1],max_value-min_value+1),device = x.device)
        x = torch.clamp(x, min=min_value, max= max_value)
        low_x = torch.floor(x)
        high_x = torch.ceil(x)
        p_low = high_x-x
        p_high = x-low_x
        index_low = low_x -min_value 
        index_high = high_x -min_value
        y.scatter_(2,index_low.long().unsqueeze(-1),p_low.unsqueeze(-1))
        y.scatter_(2,index_high.long().unsqueeze(-1),p_high.unsqueeze(-1))
        return y

    @staticmethod
    def v_r_inverse_transform(x,range_v):
        
        x = F.softmax(x, dim=1)
        sup = torch.tensor([x for x in range(range_v[0],range_v[1]+1)],device = x.device)
        value = torch.sum(x*sup,dim = 1)
        value = Transform.inverse_h_transform(value)
        return value
    @staticmethod
    def inverse_transform(x, range_v):
        assert torch.allclose(torch.sum(x, dim= 1), torch.ones((x.shape[0])))

        sup = torch.tensor([x for x in range(range_v[0],range_v[1]+1)],device = x.device)
        value = torch.sum(x*sup,dim = 1)
        value = Transform.inverse_h_transform(value)
        return value
    @staticmethod
    def transform_values(x,range_v):
        x = Transform.h_transform(x)
        x = Transform.phi_t(x,range_v[0],range_v[1])
        return x

class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""
    def __init__(self, known_bounds):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE
    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

@ray.remote
class SharedStorage(object):
    def __init__(self,conf):
        self._networks = {}
        self._net_weights = {}
        self.conf = conf
    def is_avaiable(self):
        return len(self._networks)>0
    def latest_network(self):
        return self._networks[max(self._networks.keys())]
    def latest_weights(self):
        return self._net_weights[max(self._networks.keys())]

    def save_network(self, step, network):
        network.training_steps = step
        self._networks[step] = network
        self._net_weights[step] = network.get_weights() 
        
class Player(object):
    def __init__(self, id=1):
        self.id = id

    def __eq__(self, other):
        return self.id == other.id

class Environment(object):
    """The environment MuZero is interacting with."""
    def __init__(self, env_name, render = False):
        self.env = gym.make(env_name)
        self.state_in = torch.from_numpy(self.env.reset())
        self.action_space_size = self.env.action_space.n
        self.done = False
        self.render = render
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.done = done
        state = torch.from_numpy(state)
        if self.render:
            self.env.render()
        if self.done:
            self.env.close()
        return reward, state


class ActionHistory(object):
    """Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """
    def __init__(self, history, action_space_size):
        self.history = list(history)
        self.action_space_size = action_space_size
        
    def clone(self):
        return ActionHistory(self.history, self.action_space_size)
    
    def add_action(self, action):
        self.history.append(action)
        
    def last_action(self):
        return self.history[-1]
    
    def action_space(self):
        return [i for i in range(self.action_space_size)]
    def to_play(self):
        return Player()


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, env_name ,discount, render = False):
        self.environment = Environment(env_name, render) # Game specific environment.
        self.history = []
        self.rewards = []
        self.image = [self.environment.state_in]
        self.child_visits = []
        self.root_values = []
        self.action_space_size = self.environment.action_space_size
        self.discount = discount
        
    def terminal(self):
        # Game specific termination rules.
        return self.environment.done

    def kill_game(self):
        self.environment.env.close()

    def len_history(self):
        return len(self.history)

    def get_history(self):
        return self.history
    def get_priority(self):
        return self.priority

    def legal_actions(self):
        # Game specific calculation of legal actions.
        return [i for i in range(self.action_space_size)]
    
    def apply(self, action):
        reward,state = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)
        self.image.append(state)
        
    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (index for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space 
         ])
        self.root_values.append(root.value())
    def make_image(self, state_index):
        if state_index-5>=0:
            return torch.cat(self.image[state_index+1-5:state_index+1],0).unsqueeze(0).float()
        else:
            state = torch.zeros((1,4*5)).float()
            state[0,:(state_index+1)*4] = torch.cat(self.image[:state_index+1],0).unsqueeze(0)
            return state

    
    def make_target(self, state_index, num_unroll_steps, td_steps,
        to_play: Player):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        value_target = []
        reward_target = []
        policy_target = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps 
            else:
                value = 0
            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i # pytype: disable=unsupported-operands

            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0
            if current_index < len(self.root_values):
                policy_target.append( self.child_visits[current_index])
                value_target.append(value)
                reward_target.append( last_reward)
            else:
                value_target.append(0)
                reward_target.append(last_reward)
                policy_target.append([0 for i in range(self.action_space_size)])
        return value_target,reward_target,policy_target
    def to_play(self):
        return Player()
    
    def action_history(self):
        return ActionHistory(self.history, self.action_space_size)

@ray.remote
class ReplayBuffer(object):

    def __init__(self, config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.rewards_games = []

    def get_mean_rewards(self):
        return sum(self.rewards_games)/len(self.rewards_games)
    def save_game(self, game):
        if len(self.buffer) > self.window_size:
           # self.index += 1 
            self.buffer.pop(0)
        if len(self.rewards_games)>50:
            self.rewards_games.pop(0)
        self.buffer.append(game)
        self.rewards_games.append(sum(game.rewards))
    def get_len(self):
        return len(self.buffer)
    def sample_batch(self, num_unroll_steps, td_steps):
        actions = []
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        images = torch.cat([g.make_image(i) for (g,i) in game_pos])
        for g,i in game_pos:
            action = g.get_history()[i:i+ num_unroll_steps]
            action += [0 for i in range(num_unroll_steps-len(action))]
            actions.append(torch.tensor(action))
        actions = torch.stack(actions)
        targets = [g.make_target(i, num_unroll_steps, td_steps, g.to_play()) for (g,i) in game_pos]
        target_values, target_rewards, target_policy = list(zip(*targets))
        return images, actions, (torch.Tensor(target_values), torch.Tensor(target_rewards),torch.Tensor(target_policy))

    def sample_game(self):
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[np.random.randint(0,high = len(self.buffer))] #TODO add sampling function

    def sample_position(self, game):
        # Sample position from game either uniformly or according to some priority.
        return np.random.randint(max(0,game.len_history()-20),high =game.len_history())  



