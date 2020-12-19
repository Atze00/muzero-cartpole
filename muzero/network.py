import torch
from torch import Tensor
import torch.nn as nn
import typing
from muzero.utils import Transform

class NetworkOutput(typing.NamedTuple):
    value: Tensor
    reward: Tensor
    policy_logits: Tensor
    hidden_state: Tensor
        
        

class Network(nn.Module):
    def __init__(self,action_space_size, range_v):
        super().__init__()
        self.action_space = action_space_size
        self.range_v =range_v
        self.training_steps = 0
    def initial_inference(self, image):
        # representation + prediction function
        state = self.representation(image)
        p,v = self.prediction(state)
        if not self.training:
            v = Transform.v_r_inverse_transform(v,self.range_v) 
        return NetworkOutput(v, torch.zeros(v.shape,device = p.device), p, state)
    def recurrent_inference(self, state, action):
        state,r = self.dynamic(state, action)
        p,v = self.prediction(state)        
        if not self.training:
            r = Transform.v_r_inverse_transform(r,self.range_v) 
            v = Transform.v_r_inverse_transform(v,self.range_v) 
        # dynamics + prediction function
        return NetworkOutput(v, r, p, state)
    def get_weights(self):
        # Returns the weights of this network.
        return self.state_dict()

    def representation(self, image_input):
        raise NotImplementedError

    def prediction(self):
        raise NotImplementedError
    
    def dynamic(self):
        raise NotImplementedError


