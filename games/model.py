import torch
import torch.nn as nn
from muzero.network import Network
from torch import Tensor


class PoleNet(Network):
    def __init__(self,action_space_size,range_v):
        super().__init__(action_space_size,range_v)
        self._representation =  nn.Sequential(
                            nn.Linear(4*5, 64),
                            nn.ReLU(),
                            )
        self._prediction_a = nn.Sequential(
                            nn.Linear(64, 80),
                            nn.ReLU(),
                            nn.Linear(80, self.action_space),
                            )
        self._prediction_v = nn.Sequential(
                            nn.Linear(64, 80),
                            nn.ReLU(),
                            nn.Linear(80, 21),
                            )
        self._dynamic_s = nn.Sequential(
                            nn.Linear(64+2, 80),
                            nn.ReLU(),
                            nn.Linear(80, 64),
                            nn.ReLU(),
                            )    
        self._dynamic_r = nn.Sequential(
                            nn.Linear(64+2, 80),
                            nn.ReLU(),
                            nn.Linear(80, 21),
                            )
    def representation(self, image_input):
        return self._representation(image_input)

    def prediction(self, state):
        a = self._prediction_a(state)
        v =self._prediction_v(state)
        return a, v
    def dynamic(self, state, action):
        if len(action.shape)==0:
            action_encoded = torch.zeros(1,self.action_space, dtype = torch.float32, device = action.device)
            action_encoded.scatter_(1,action.unsqueeze(0).unsqueeze(0),value = 1)
            input_dynamic = torch.cat((state,action_encoded), dim = 1)
        else:
            assert action.shape[1]==1
            action_encoded = torch.zeros(action.shape[0],self.action_space, dtype = torch.float32, device = action.device)
            action_encoded.scatter_(1,action,value = 1)
            input_dynamic = torch.cat((state,action_encoded), dim = 1)
        s = self._dynamic_s(input_dynamic)
        r = self._dynamic_r(input_dynamic)

        return s,r
