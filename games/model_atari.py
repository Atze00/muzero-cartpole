from __future__ import annotations
import torch
import torch.nn as nn
from muzero.network import Network
from torch import Tensor

class DynamicNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, n_residual=3):
        super().__init__()
        assert n_residual >= 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res1 = nn.Sequential(*(ResidualBlock(self.in_channels) for i in range(n_residual//3)))
        self.conv1 =  nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3,
                                      stride=1, bias=False,padding = 1)
        self.reward = nn.Sequential(                    
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(),
                    nn.Conv2d(self.out_channels, self.out_channels,
                              kernel_size=3,
                              stride=1, bias=False,padding = 1),
                    )
        self.fc_r = nn.Linear(6*6*self.out_channels, 1)
        #self.res2 = nn.Sequential(*[ResidualBlock(self.out_channels) for i in range(n_residual-(n_residual//3)-1)])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.res1(x)
        x = self.conv1(x)
        r = self.reward(x)
        r = r.view(r.shape[0],-1)
        r = self.fc_r(r)
        #x = self.res2(x)
        return x,r
    
    
class PredictionNetwork(nn.Module):
    def __init__(self,shape, in_channels, reduction_coeff, action_space):
        super().__init__()
        self.reduction_coeff= reduction_coeff
        kernel_size= 3
        self.action_space = action_space
        stride = 1
        self.in_channels= in_channels
        self.conv1 = nn.Sequential(
                            nn.Conv2d(self.in_channels, self.in_channels//reduction_coeff,
                                      kernel_size=kernel_size,
                                      stride=stride, bias=False,padding = 1),
                            nn.BatchNorm2d(self.in_channels//reduction_coeff),
                            nn.ReLU()
                            )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(self.in_channels//reduction_coeff, self.in_channels//(reduction_coeff*2),
                              kernel_size=kernel_size,
                              stride=stride, bias=False,padding = 1),
                    )
        self.fc = nn.Linear(shape[0]*shape[1]*self.in_channels//(reduction_coeff*2),action_space+1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x= self.conv1(x)
        x= self.conv2(x)
        x= x.view(x.shape[0],-1)
        x= self.fc(x)
        return x[:,:self.action_space], x[:,self.action_space:self.action_space+1]
        
        
class RepresentationNetwork(nn.Module):
    def __init__(self, in_channels,out_channels=None ):
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = in_channels *2
        else:
            self.out_channels = out_channels
        kernel_size= 3
        stride = 2
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size,
                                      stride=stride, bias=False,padding = 1)
        self.res1 = nn.Sequential( ResidualBlock(self.in_channels),ResidualBlock(self.in_channels) )
        
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size,
                                          stride=stride, bias=False,padding = 1)
                            
        self.res2 = nn.Sequential( ResidualBlock(self.out_channels),ResidualBlock(self.out_channels),
                                 ResidualBlock(self.out_channels))
        
        self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2,padding = 1)
        
        self.res3 = nn.Sequential( ResidualBlock(self.out_channels),ResidualBlock(self.out_channels),
                                 ResidualBlock(self.out_channels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self,x):
        x= self.conv1(x)
        #x= self.res1(x)
        x= self.conv2(x)
        x= self.res2(x)
        x= self.avgpool(x)
        x= self.res3(x)
        x= self.avgpool(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.block = nn.Sequential(
#                nn.BatchNorm2d(self.channels),
#                nn.ReLU(),#TODO real model
#                nn.Conv2d(self.channels, self.channels, kernel_size=3,
#                              stride=1, bias=False, padding = 1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3,
                              stride=1, bias=False, padding = 1)
                    )

    def forward(self,x):
        identity = x
        x= self.block(x)
        return x + identity



class MuZeroAtari(Network):
    def __init__(self,action_space_size):
        super().__init__(action_space_size)
        self._representation = RepresentationNetwork(128,16)
        self._prediction = PredictionNetwork(shape = (6,6), in_channels = 16, reduction_coeff = 2 , action_space = self.action_space)#18
        self._dynamic = DynamicNetwork(16+self.action_space, 16)

    def representation(self, image_input):
        return self._representation(image_input)

    def prediction(self, state):
        return self._prediction(state)

    def dynamic(self, state, action):
        if len(action.shape)==0:
            action_encoded = torch.zeros(1,self.action_space, dtype = torch.float32, device = action.device)
            action_encoded.scatter_(1,action.unsqueeze(0).unsqueeze(0),value = 1)
            action_encoded = action_encoded.unsqueeze(-1).unsqueeze(-1).expand(1,self.action_space,6,6)
            input_dynamic = torch.cat((state,action_encoded), dim = 1)
        else:
            assert action.shape[1]==1
            action_encoded = torch.zeros(action.shape[0],self.action_space, dtype = torch.float32, device = action.device)
            action_encoded.scatter_(1,action,value = 1)
            action_encoded = action_encoded.unsqueeze(-1).unsqueeze(-1).expand(action.shape[0],self.action_space,6,6)
            input_dynamic = torch.cat((state,action_encoded), dim = 1)

        return self._dynamic(input_dynamic)
