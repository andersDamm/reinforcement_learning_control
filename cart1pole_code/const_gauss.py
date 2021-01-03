import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import dmc2gym
import copy

if torch.cuda.is_available():
    print('cuda is available')

# initialize environment
seed_env=16
env = dmc2gym.make(domain_name='cartpole', task_name='balance',seed=seed_env)
env.seed(seed_env)

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 200
buffer_limit = int(1e6) #1000000

tau          = 0.005 # for target network soft update
num_episodes = 1000

obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]

# critic network input/output dimension
n_inputs = obs_size
n_outputs = 1

# actor network input/output dimension
m_inputs = obs_size
m_outputs = act_size

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

def param_noise(net,std):
    
    with torch.no_grad():
        for param in net.parameters():
            param.add_(torch.randn(param.size()) * std)


class MuNet(nn.Module):
    
    def __init__(self, n, o, learning_rate):
        super(MuNet, self).__init__()
        # network
                
        self.fc1 = nn.Linear(5, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mu = nn.Linear(300, 1)            
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu


class QNet(nn.Module):
    
    def __init__(self, n, m, o, learning_rate):
        super(QNet, self).__init__()
        # network
                
        self.fc_s = nn.Linear(5, 200)
        self.fc_a = nn.Linear(1,200)
        self.fc_q = nn.Linear(400, 300)
        self.fc_out = nn.Linear(300,1)
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x, a):
                            
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train(mu,mu_target,q,q_target,memory):
    ss,aa,rr,ss1,ddone  = memory.sample(batch_size)
    
    target = rr + gamma*q_target(ss1,mu_target(ss1)) * ddone
    q_loss = F.mse_loss(q(ss,aa),target.detach())
    q.optimizer.zero_grad()
    q_loss.backward()
    q.optimizer.step()
    mu_loss = -q(ss,mu(ss)).mean()
    mu.optimizer.zero_grad()
    mu_loss.backward()
    mu.optimizer.step()
    

def trainloop():

    env = dmc2gym.make(domain_name='cartpole', task_name='balance',seed=seed_env)
    env.seed(seed_env)

    q = QNet(n_inputs, m_outputs, n_outputs, lr_q)
    q_target = QNet(n_inputs, m_outputs, n_outputs, lr_q)

    mu = MuNet(m_inputs, m_outputs, lr_mu)
    mu_target = MuNet(m_inputs, m_outputs, lr_mu)

    memory = ReplayBuffer()
    score = 0
    score_test = 0
    std_dev = 0.2

    sigma = 0.1

    #ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    for i in range(num_episodes):

        s = env.reset()
        done = False
                
        while not done:

            with torch.no_grad():
                a = mu(torch.from_numpy(s).float()) 
            
            a = a + torch.randn(1)*sigma
            a = torch.clamp(a, min = -1, max = 1)

            s_prime, r, done, info = env.step(a.data.numpy())
            
            if done:
                break
            
            memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime

            if memory.size() > 2000:
                train(mu,mu_target,q,q_target,memory)
                soft_update(mu,mu_target)
                soft_update(q,q_target)

        if i%1==0 and i!=0:
            for _ in range(10):
                s = env.reset()
                done = False
                while not done:
                    
                    with torch.no_grad():
                        a = mu(torch.from_numpy(s).float())
                    s, r, done, info = env.step(a.data.numpy())
 
                    if done:
                        break
 
                    score_test += r        
                
            if i == 1:
                print("# of episode, training score, validation score, sigma",flush = True)

            print("{}, {:.1f}, {:.1f}, {:.5f}".format(i, score, score_test/10, sigma),flush = True)            
            
            score = 0.0
            score_test = 0.0


    env.close()
    
    return mu
     
mu_trained = trainloop()

path = "/zhome/0b/b/108018/deep_learning/cartpole/out/constant_gaussian/actor_net"

torch.save(mu_trained.state_dict(), path)



