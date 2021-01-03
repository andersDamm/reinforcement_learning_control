from dm_control import suite
import numpy as np
from PIL import Image
import subprocess
import torch
seed = 0
env = suite.load(domain_name='cartpole', task_name="balance", task_kwargs={'random': seed})

action_spec = env.action_spec()

time_step_counter = 0

subprocess.call(['rm','-rf','frames'])
subprocess.call(['mkdir','-p','frames'])
s = env.reset()


env._physics.get_state()
# K_LQR = torch.tensor([[ -0.095211883797698 , 23.498594950851146  ,-0.506162305244223 ,  5.042039423490390]]) # this is K
# K_LQR = torch.tensor([[-0.095010210894877, 22.995971797251055, -0.495875655825119, 4.714827734408781]]) # this is K
K_LQR = torch.tensor([[-0.0952, 23.1348, -0.4992, 5.0009]]) # this is K

Q = np.array([[10, 0, 0, 0],[0, 1000, 0, 0],[0, 0, 10, 0],[0, 0, 0, 10]])
R = np.array([1000])

Reward = 0
J = 0

num_episodes = 2000

for i in range(num_episodes):

    print(i)

    time_step   = env.reset()

    while not time_step.last():
    
        States = env._physics.get_state()
        
        u_lqr = torch.matmul(K_LQR,torch.from_numpy(States).float())
        action = np.random.uniform(action_spec.minimum,
                                    action_spec.maximum,
                                    size = action_spec.shape)
        u_lqr = torch.clamp(u_lqr, min = -1, max = 1)                             
        time_step = env.step(u_lqr)
        
        J = J + np.array([u_lqr*R*u_lqr]) + np.matmul(np.array([States]),np.matmul(Q,np.array([States]).T))
        Reward = Reward + time_step.reward


print(Reward/num_episodes)
print(J/num_episodes)