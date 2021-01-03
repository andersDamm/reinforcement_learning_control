import gym
import random
import collections
import numpy as np
import torch
from PIL import Image
import subprocess
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import dmc2gym
from dm_control import suite
seed = 0
env = suite.load(domain_name='cartpole', task_name="balance", task_kwargs={'random': seed})

action_spec = env.action_spec()
time_step   = env.reset()

def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)
class MuNet(nn.Module):
    
    def __init__(self):
        super(MuNet, self).__init__()
        # network
                
        self.fc1 = nn.Linear(5, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mu = nn.Linear(300, 1)          
        # training
        #self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu



time_step_counter = 0



def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs
# actor network input/output dimension
subprocess.call(['rm','-rf','frames'])
subprocess.call(['mkdir','-p','frames'])
s = env.reset()

env._physics.get_state()
mu = MuNet()
K_LQR = torch.tensor([[ -0.095211883797698 , 23.498594950851146  ,-0.506162305244223 ,  5.042039423490390]]) # this is K

while not time_step.last() and time_step_counter <1000:
  
  States = env._physics.get_state()
  
  u_lqr = torch.matmul(-K_LQR,torch.from_numpy(States).float())
  action = np.random.uniform(action_spec.minimum,
                              action_spec.maximum,
                              size = action_spec.shape)
  u_lqr = torch.clamp(u_lqr, min = -1, max = 1)                             
  time_step = env.step(u_lqr)

 
  print(States)
  image_data = env.physics.render(height=480,width=480, camera_id=0)
  img = Image.fromarray(image_data,'RGB')
  img.save("frames/frame-%.10d.png" % time_step_counter)
  time_step_counter += 1
subprocess.call([
  'ffmpeg','-framerate', '50', '-y', '-i','frames/frame-%010d.png','-r','30','-pix_fmt','yuv420p','cartpole_LQR.mp4'
])

