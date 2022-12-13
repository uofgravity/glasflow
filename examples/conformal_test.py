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
from torch import tensor as t
matplotlib.style.use('ggplot')

#get_ipython().run_line_magic('matplotlib', 'wx')
plt.ion()
import h5py

# Create some training data on the surface of a sphere

# In[2]:


import scipy.stats
from scipy.stats import norm, uniform


from glasflow.flows.conformal import ConformalFlow
from glasflow.transforms import conformal
from glasflow.nflows import transforms
from glasflow.nflows import distributions
from glasflow.nflows.transforms.base import Transform


class SphereEmbedding(Transform):
    """
    Transforms coordinates between polar angles and
    catesian cordinates in a space of one higher dimension
    Parameters:
        - dim: int. Dimension of the input
    """
    def __init__(self, dim: int = 2):
        self.dim = dim
        super().__init__()
    
    def forward(self, inputs, context=None):
        """
        Forward transformation theta -> x

        Args:
            theta (torch.Tensor): polar coordinates

        Yields:
            x  (torch.Tensor): Euclidean coordinates
        """
        cos = torch.cos
        sin = torch.sin
        in_shape = inputs.shape
        print('Forward input shape',in_shape)

        out_shape = list(in_shape)
        out_shape[-1] = self.dim+1
        assert self.dim == in_shape[-1]
        #theta = inputs.view(in_shape[0], -1)
        outputs = inputs.new_empty(out_shape)
        outputs[:,0] = sin(inputs[:,0])*cos(inputs[:,1])
        outputs[:,1] = sin(inputs[:,0])*sin(inputs[:,1])
        outputs[:,2] = cos(inputs[:,0])
        assert torch.all(~torch.isnan(outputs))

        logabsdet = inputs.new_zeros(in_shape[0])
        
        return outputs, logabsdet
    
    def inverse(self, inputs, context=None):
        in_shape = inputs.shape
        print('Inverse input shape',in_shape)
        out_shape = list(in_shape)
        out_shape[-1] = self.dim
        assert self.dim == in_shape[-1]-1
        outputs = inputs.new_empty(out_shape)
        r = torch.sqrt(torch.sum(inputs[:,:]**2, axis=-1))
        print(r.shape)
        outputs[:,0] = torch.acos(inputs[:,2]/r)
        outputs[:,1] = torch.atan2( inputs[:,1], inputs[:,0])
        assert torch.all(~torch.isnan(outputs))
        logabsdet = inputs.new_zeros(in_shape[0])
        return outputs, logabsdet

class SphereCEFlow(ConformalFlow):
    def __init__(self, dim, base_distribution = None):
        n = dim+1
        m = dim
        if not base_distribution:
            base_distribution = RealNVP(
                 n_inputs = dim,
                 n_transforms = 4,
                 n_neurons = 32)

        conf_transform = transforms.CompositeTransform([
            SphereEmbedding(2),
            conformal.ConformalScaleShift(n, m),
            conformal.Orthogonal(n),
            conformal.SpecialConformal(n, m),
            conformal.Pad(n, m),
        ])
        

        super().__init__(conf_transform, distribution=base_distribution)


from scipy.stats import halfnorm

#################################
dist = halfnorm()
x = halfnorm.rvs(size=(500, 2))


##################################
## Load event samples
t = lambda x: torch.Tensor(x)[None,:]
filename = '/home/john/work/O3/O3a_PE_repos/o3a_catalog_events/S190513bm/Preferred/PESummary_metafile/posterior_samples.h5'
with h5py.File(filename,'r') as posfile:
    samps = posfile['C01:IMRPhenomPv2']['posterior_samples']
    ra, dec, phase, psi, iota =   samps['ra'], samps['dec'], samps['phase'], samps['psi'], samps['iota']
    x = torch.cat(( torch.pi/2 - t(dec), t(ra)
                   #t(phase),
                   #t(psi),
                   #t(iota)
                   )
                  ).T


sph_data = torch.FloatTensor(x)

plt.plot(x[:, 1], x[:, 0],'x')
plt.show()

spflow = SphereCEFlow(2) #, base_distribution = distributions.StandardNormal([2]))

print("Init")

with torch.no_grad():
    gen_samples = spflow.sample(1000)
    sample_mid_latent, _ = spflow._transform.forward(sph_data)
    sample_recons, _ =  spflow._transform.inverse(sample_mid_latent)


import torch.optim as opt
from torch.utils.data import DataLoader
batch_size = 1000
optim = opt.Adam(spflow.parameters(), lr=0.01)
scheduler = opt.lr_scheduler.MultiStepLR(optim, milestones=[40], gamma=0.5)
NTRAIN = 500
def schedule():
    '''Yield epoch weights for likelihood and recon loss, respectively'''
    for _ in range(NTRAIN):
        yield 10, 10000
        scheduler.step()
        
loader = DataLoader(sph_data, batch_size=batch_size, shuffle=True, num_workers=4)


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

        #zz = mid_latent.detach()
        if batch==0:
            zz = mid_latent.detach()
            ax.clear()
            ax.plot(zz[:, 0], zz[:, 1],'x')
            #plt.draw()
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


#fig = plt.figure()
#plt.scatter(sample_mid_latent[:, 0], sample_mid_latent[:, 1])


with torch.no_grad():
    gen_samples = spflow.sample(1000).detach()

fig = plt.figure()
ax = fig.gca()
ax.plot(gen_samples[:, 0], gen_samples[:, 1],'x')


# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')
# point_plot = ax.scatter(gen_samples[:,0], gen_samples[:,1], gen_samples[:,2], color='#faab36')
# ax.auto_scale_xyz([-1.3, 1.3], [-1.3, 1.3], [-1, 1]) # Correct aspect ratio manually
# ax.view_init(elev=20, azim=260)
