from muzero.conf import MuZeroConfig
from .model import PoleNet


def make_pole_config(env_name):
    training_steps_n = 56000
    def visit_softmax_temperature(num_moves, training_steps):
#        if training_steps < training_steps_n//2:
#            return 1.0
#        elif training_steps < training_steps_n//4*3:
#            return 0.5
#        else:
#            return 0.25
        return 1.0
    return MuZeroConfig(
            action_space_size=2,
            max_moves=600, 
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=50,
            batch_size=124,
            td_steps=20,
            num_actors=35,#350 standard
            lr_init=0.5,
            policy_w= 0.5,
            lr_decay_steps=[350e3],
            visit_softmax_temperature_fn=visit_softmax_temperature,
            Network = PoleNet, 
            training_steps = training_steps_n,
            range_v =(-10,10), 
            env_name=env_name)

