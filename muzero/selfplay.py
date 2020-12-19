import torch
import ray
import math
import numpy as np
from muzero.utils import MinMaxStats

    

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
@ray.remote
def run_selfplay(config, storage, replay_buffer):
    network = config.make_uniform_network()
    while True:
        if ray.get(storage.is_avaiable.remote()):
            network.load_state_dict( ray.get(storage.latest_weights.remote()))
        with torch.no_grad():
            game = play_game(config, network.eval())
        replay_buffer.save_game.remote(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config, network, render = False):
    game = config.new_game(render = render)
    while not game.terminal() and game.len_history() < config.max_moves:

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.

        index = game.len_history()
        root = Node(0)
        current_observation = game.make_image(index)
        network_output = network.initial_inference(current_observation)
        expand_node(root, game.to_play(), game.legal_actions(), network_output )
        add_exploration_noise(config, root)
        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, game.len_history(), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    game.kill_game()
    return game

class Node(object):
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config, root, action_history, network):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,torch.tensor(history.last_action()))
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(),
                                    config.discount, min_max_stats)    

def select_action(config, num_moves, node, network):
    visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
            num_moves=num_moves, training_steps=network.training_steps)
    action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(config, node,min_max_stats):
    ubc_scores = []
    actions = []
    childs = []
    for action, child in node.children.items():
        ubc_scores.append(ucb_score(config, node, child, min_max_stats))
        actions.append(action)
        childs.append(child)

    index = np.argmax(np.asarray(ubc_scores))
    return actions[index], childs[index]


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config, parent, child,min_max_stats):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = child.reward + config.discount * min_max_stats.normalize(
                child.value())
    else:
        value_score = 0
    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node, to_play, actions,network_output):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[:,a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, to_play,discount, min_max_stats):
    for node in reversed(search_path): 
        #assert node.to_play == to_play 
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config, node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac




def softmax_sample(visit_count, t):
    visit_counts, actions = list(zip(*visit_count)) 
    y = np.asarray(visit_counts)**(1/t)
    y = y/np.sum(y)
    index = np.random.choice(len(y),p=y) 
    return actions[index]


        
