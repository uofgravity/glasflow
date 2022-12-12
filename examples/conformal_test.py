#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glasflow.flows import RealNVP
import numpy as np
import matplotlib
import seaborn as sns
from numpy import sin, cos
from matplotlib import pyplot as plt
import torch
matplotlib.style.use('ggplot')

#get_ipython().run_line_magic('matplotlib', 'wx')
plt.ion()


# Create some training data on the surface of a sphere

# In[2]:


import scipy.stats
from scipy.stats import norm, uniform


from glasflow.flows.conformal import ConformalFlow
from glasflow.transforms import conformal
from glasflow.nflows import transforms
from glasflow.nflows import distributions


class SphereCEFlow(ConformalFlow):
    def __init__(self, n_transforms: int, n_neurons: int = 32):
        n = 2
        m = 2

        conf_transform = transforms.CompositeTransform([
            conformal.ConformalScaleShift(n, m),
            conformal.Orthogonal(n),
            conformal.SpecialConformal(n, m),
            conformal.Pad(n, m),
        ])
        # base_flow = RealNVP(
        #     n_inputs=m,
        #     n_transforms=n_transforms,
        #     n_neurons=n_neurons,
        # )
        base_flow = distributions.StandardNormal([2])

        super().__init__(conf_transform, distribution=base_flow)


# In[3]:


# theta = norm(np.pi/2,0.1).rvs(1000)
# phi = uniform(0,2*np.pi).rvs(1000)
# r = 1.0

# x = r*sin(theta)*cos(phi)
# y = r*sin(theta)*sin(phi)
# z = r*cos(theta)
# x, y, z = map(lambda x: torch.tensor(x[None,:],dtype=torch.float32), (x,y,z))
# sph_data = torch.cat([x,y,z]).T
# print(sph_data.shape)

from scipy.stats import halfnorm

dist = halfnorm()
x = halfnorm.rvs(size=(1000, 2))

plt.scatter(x[: 0], x[:, 1])
plt.show()

sph_data = torch.rand(1000, 2) + 100
sph_data = torch.FloatTensor(x)


# In[4]:


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlim([-1,1])
# ax.set_ylim([-1,1])
# ax.set_zlim([-1,1])
# ax.scatter(x,y,z)
# plt.show()


# In[5]:


# from glasflow.flows.conformal import SphereCEFlow


# In[6]:


spflow = SphereCEFlow(n_transforms=2)


# In[7]:


recon = spflow.reconstruct(sph_data)


# In[8]:


x,logJ = spflow.forward(sph_data)


# In[9]:


sph_data


# In[10]:


spflow


# In[11]:


pad = spflow._transform._transforms[3]


# In[12]:


pad(sph_data)[0] - sph_data[:,:2]


# In[13]:


spflow._transform._transforms[3]

print("Init")

with torch.no_grad():
    gen_samples = spflow.sample(1000)
    sample_mid_latent, _ = spflow._transform.forward(sph_data)
    sample_recons, _ =  spflow._transform.inverse(sample_mid_latent)


# In[14]:


import torch.optim as opt
from torch.utils.data import DataLoader
batch_size = 1000
optim = opt.Adam(spflow.parameters(), lr=0.1)
scheduler = opt.lr_scheduler.MultiStepLR(optim, milestones=[40], gamma=0.5)

def schedule():
    '''Yield epoch weights for likelihood and recon loss, respectively'''
    for _ in range(45):
        yield 10, 1
        scheduler.step()
        
loader = DataLoader(sph_data, batch_size=batch_size, shuffle=True, num_workers=2)


# In[15]:


import torch
import torch.nn as nn
from tqdm import tqdm

fig = plt.figure()
ax = plt.gca()
plt.show()

for epoch, (alpha, beta) in enumerate(schedule()):
    
    # Train for one epoch
    spflow.train()
    progress_bar = tqdm(enumerate(loader))
    
    for batch, point in progress_bar:
        optim.zero_grad()
        # print(point.shape)
        # Compute reconstruction error
        with torch.set_grad_enabled(beta > 0):
            mid_latent, _ = spflow._transform.forward(point)
            reconstruction, log_conf_det = spflow._transform.inverse(mid_latent)
            reconstruction_error = torch.mean((point - reconstruction)**2)

        # Compute log likelihood
        with torch.set_grad_enabled(alpha > 0):
            log_pu = spflow._distribution.log_prob(mid_latent)
            log_likelihood = torch.mean(log_pu - log_conf_det)

        # Training step
        loss = - alpha*log_likelihood + beta*reconstruction_error
        loss.backward()
        optim.step()

        # Display results
        progress_bar.set_description(f'[E{epoch} B{batch}] | loss: {loss: 6.2f} | LL: {log_likelihood:6.2f} '
                                     f'| recon: {reconstruction_error:6.5f} ')

        zz = mid_latent.detach()
        ax.clear()
        ax.scatter(zz[:, 0], zz[:, 1])
        plt.draw()
        plt.pause(0.01)

# with torch.no_grad():
#     sample_mid_latent, _ = spflow._transform.forward(sph_data)
#     sample_recons, _ =  spflow._transform.inverse(sample_mid_latent)

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')
# point_plot = ax.scatter(sph_data[:,0], sph_data[:,1], sph_data[:,2], color='#faab36')
# recon_plot = ax.scatter(sample_recons[:,0], sample_recons[:,1], sample_recons[:,2], 
#                         color='#249ea0')
# ax.auto_scale_xyz([-1.3, 1.3], [-1.3, 1.3], [-1, 1]) # Correct aspect ratio manually
# ax.view_init(elev=20, azim=260)


fig = plt.figure()
plt.scatter(sample_mid_latent[:, 0], sample_mid_latent[:, 1])


with torch.no_grad():
    gen_samples = spflow.sample(1000).detach()

plt.scatter(gen_samples[:, 0], gen_samples[:, 1])


# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')
# point_plot = ax.scatter(gen_samples[:,0], gen_samples[:,1], gen_samples[:,2], color='#faab36')
# ax.auto_scale_xyz([-1.3, 1.3], [-1.3, 1.3], [-1, 1]) # Correct aspect ratio manually
# ax.view_init(elev=20, azim=260)

# In[ ]:





# In[ ]:




