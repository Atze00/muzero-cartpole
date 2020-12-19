from muzero.utils import Game
import collections

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MuZeroConfig(object):
    def __init__(self,
            action_space_size,
            max_moves,
            discount,
            dirichlet_alpha,
            num_simulations,
            batch_size,
            td_steps,
            num_actors,
            lr_init,
            lr_decay_steps,
            visit_softmax_temperature_fn,
            env_name,
            Network,
            training_steps,
            range_v = (-60,60),
            known_bounds = None):
        ### Self-Play
        self.env_name = env_name
        self.action_space_size = action_space_size
        self.num_actors = num_actors
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount
        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds
        ### Training
        self.training_steps =training_steps 
        self.checkpoint_interval = int(2000)
        self.window_size = int(500)#1e6
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.range_v = range_v

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps
        def make_uniform_network():
            return Network(action_space_size,self.range_v)
        self.make_uniform_network = make_uniform_network
    def new_game(self, render= False):
        return Game(self.env_name,self.discount,render)

