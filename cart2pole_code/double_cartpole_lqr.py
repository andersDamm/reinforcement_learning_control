from dm_control import suite
import numpy as np
from PIL import Image
import subprocess
import torch
seed = 0
env = suite.load(domain_name='cartpole', task_name="two_poles", task_kwargs={'random': seed})

action_spec = env.action_spec()

time_step_counter = 0

subprocess.call(['rm','-rf','frames'])
subprocess.call(['mkdir','-p','frames'])
s = env.reset()

env._physics.get_state()
# K_LQR = torch.tensor([[ -0.095211883797698 , 23.498594950851146  ,-0.506162305244223 ,  5.042039423490390]]) # this is K
# K_LQR = torch.tensor([[0.0880, -137.0451, 139.2033, 0.5070, -10.6144, 19.7211]]) # this is K
K_LQR = torch.tensor([[0.0670, -115.3641, 112.3498, 0.3516, -2.9878, 4.9511]])

R = 0

time_step   = env.reset()

States = env._physics.get_state()
print(States)

while not time_step.last():

    States = env._physics.get_state()
    
    u_lqr = -torch.matmul(K_LQR,torch.from_numpy(States).float())
    action = np.random.uniform(action_spec.minimum,
                                action_spec.maximum,
                                size = action_spec.shape)
    #u_lqr = torch.clamp(u_lqr, min = -1, max = 1)                             
    time_step = env.step(u_lqr)

    # print(States)

    time_step_counter = time_step_counter + 1

    R = R + time_step.reward

print(States)
print(R)
    