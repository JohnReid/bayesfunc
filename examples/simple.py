#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from operator import attrgetter
import logging
import torch as t
from torch.distributions import Normal
import torch.nn as nn
import bayesfunc as bf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload
reload(bf)

logging.basicConfig(level=logging.INFO)
mpl.rcParams['backend']
mpl.use('qt5agg')
plt.ion()
sns.set_theme()


class ActivationSpy:
    """A forward hook to spy on activations."""
    def __call__(self, module, input_, output):
        self.activations = output


def schur_complement(K, *dims):
    # upper = False
    # chol = t.cholesky(K.ii, upper=upper, *dims)
    # a = t.cholesky_solve(K.it, chol, upper=upper, *dims)
    invii = t.pinverse(K.ii)
    Kti = t.transpose(K.it, -1, -2)
    term1 = t.matmul(invii, K.it)
    term2 = t.matmul(Kti, term1)
    return t.diag(K.tt) - term2


def calculate_elbo(net, samples):
    output, logpq, _ = bf.propagate(net, X.expand(samples, -1, -1))
    ll = Normal(output, 3 / scale).log_prob(y).sum(-1).mean(-1)
    assert ll.shape == (samples,)
    assert logpq.shape == (samples,)
    return ll + logpq / train_batch


def train(net, samples=10):
    opt = t.optim.Adam(net.parameters(), lr=0.05)
    for i in range(2000):
        opt.zero_grad()
        elbo = calculate_elbo(net, samples=samples)
        (-elbo.mean()).backward()
        opt.step()


def plot(net):
    with t.no_grad():
        xs = t.linspace(-6, 6, 100)[:, None].to(device=device, dtype=dtype)
        # set sample=100, so we draw 100 different functions
        #ys, _, _ = bf.propagate(net, xs.expand(100, -1, -1))
        ys = net(xs.expand(100, -1, -1))
        mean_ys = ys.mean(0)
        std_ys = ys.std(0)
        plt.fill_between(xs[:, 0], mean_ys[:, 0] - 2 * std_ys[:, 0], mean_ys[:, 0] + 2 * std_ys[:, 0], alpha=0.5)
        plt.plot(xs, mean_ys)
        plt.scatter(X, y, c='r')


# Generate data
in_features = 1
out_features = 1
train_batch = 40

t.manual_seed(0)
X = t.zeros(train_batch, in_features)
X[:int(train_batch / 2), :] = t.rand(int(train_batch / 2), in_features) * 2. - 4.
X[int(train_batch / 2):, :] = t.rand(int(train_batch / 2), in_features) * 2. + 2.
y = X**3. + 3 * t.randn(train_batch, in_features)

# Rescale the outputs to have unit variance
scale = y.std()
y = y / scale

dtype = t.float64
device = "cpu"

X = X.to(dtype=dtype, device=device)
y = y.to(dtype=dtype, device=device)

plt.scatter(X, y)


## Mean-field variational inference
## ======

#net = nn.Sequential(
#    bf.FactorisedLinear(in_features=1, out_features=50, bias=True),
#    nn.ReLU(),
#    bf.FactorisedLinear(in_features=50, out_features=50, bias=True),
#    nn.ReLU(),
#    bf.FactorisedLinear(in_features=50, out_features=1, bias=True)
#)
#net = net.to(device=device, dtype=dtype)
#train(net)
#plot(net)


## Local-inducing with Bayesian neural networks
## =======

#net = nn.Sequential(
#    bf.LILinear(in_features=1, out_features=50, bias=True),
#    nn.ReLU(),
#    bf.LILinear(in_features=50, out_features=50, bias=True),
#    nn.ReLU(),
#    bf.LILinear(in_features=50, out_features=1, bias=True)
#)
#net = net.to(device=device, dtype=dtype)
#train(net)
#plot(net)


## Global-inducing with Bayesian neural networks
## =====

#inducing_batch = 40
#net = nn.Sequential(
#    bf.GILinear(in_features=1, out_features=50, inducing_batch=inducing_batch, bias=True),
#    nn.ReLU(),
#    bf.GILinear(in_features=50, out_features=50, inducing_batch=inducing_batch, bias=True),
#    nn.ReLU(),
#    bf.GILinear(in_features=50, out_features=1, inducing_batch=inducing_batch, bias=True)
#)
#net = bf.InducingWrapper(net, inducing_batch=inducing_batch, inducing_data=t.linspace(-4, 4, inducing_batch)[:, None])
#net = net.to(device=device, dtype=dtype)
#train(net)
#plot(net)


## Local-inducing with Gaussian processes
## ========
##
## KernelLIGP comes packaged with a squared-exponential kernel.
##
## Note that typically in such nets, we'd use skip connections to stabilise learning.  If we don't include them, things can become quite unstable (try changing the random seed...)

#t.manual_seed(0)
#inducing_batch = 40
#net = nn.Sequential(
#    bf.KernelLIGP(in_features=1, out_features=10, inducing_batch=inducing_batch),
#    bf.KernelLIGP(in_features=10, out_features=1, inducing_batch=inducing_batch)
#)
#net = net.to(device=device, dtype=dtype)

#train(net)
#plot(net)


## Global-inducing deep Gaussian processes
## =====

#inducing_batch = 40
#net = nn.Sequential(
#    bf.SqExpKernel(in_features=1, inducing_batch=inducing_batch),
#    bf.GIGP(out_features=10, inducing_batch=inducing_batch),
#    bf.SqExpKernel(in_features=10, inducing_batch=inducing_batch),
#    bf.GIGP(out_features=1, inducing_batch=inducing_batch)
#)
#net = bf.InducingWrapper(net, inducing_batch=inducing_batch, inducing_data=t.linspace(-5, 5, inducing_batch)[:, None])
#net = net.to(device=device, dtype=dtype)

#train(net)
#plot(net)


## Inducing NNGP
## ===============

#inducing_batch = 40
#net = nn.Sequential(
#    bf.FeaturesToKernel(inducing_batch=inducing_batch),
#    bf.SqExpKernelGram(),
#    bf.SqExpKernelGram(),
#    bf.GIGP(out_features=1, inducing_batch=inducing_batch)
#)
#net = bf.InducingWrapper(net, inducing_batch=inducing_batch, inducing_shape=(inducing_batch, 1))
#net = net.to(device=device, dtype=dtype)

#train(net)
#plot(net)


# Deep kernel processes
# ==========

inducing_batch = 37
net = nn.Sequential(
    bf.FeaturesToKernel(inducing_batch=inducing_batch),
    bf.SqExpKernelGram(),
    bf.IWLayer(inducing_batch),
    bf.SqExpKernelGram(),
    bf.GIGP(out_features=1, inducing_batch=inducing_batch)
)
net = bf.InducingWrapper(net, inducing_batch=inducing_batch, inducing_data=t.linspace(-5, 5, inducing_batch)[:, None])
net = net.to(device=device, dtype=dtype)

train(net)
plot(net)

# Examine network's parameters
net
feat2kernel = net[1][0]
sqexp1 = net[1][1]
iw_layer = net[1][2]
sqexp2 = net[1][3]
gp = net[1][4]
list(feat2kernel.named_parameters())
list(sqexp1.named_parameters())
list(iw_layer.named_parameters())
list(sqexp2.named_parameters())
list(gp.named_parameters())

# No learning going on for V?
iw_layer.V.sum()
iw_layer.V.max()
iw_layer.V.min()

# Spy activations
spies = [ActivationSpy() for _ in net[1]]
hooks = [module.register_forward_hook(spy) for module, spy in zip(net[1], spies)]
ys = net(X.expand(100, X.shape[0], -1))
list(map(type, map(attrgetter('activations'), spies)))

# Plot Gram matrices
show_schur = False
nrow, ncol = 3, len(spies) - 1  # Last layer does not output Gram matrix
fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(12, 8))
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        if show_schur:
            gram = schur_complement(spies[j].activations)
        else:
            gram = spies[j].activations.ii
        sns.heatmap(gram[i].detach(), ax=ax, xticklabels=False, yticklabels=False)

fig.show()

plt.close(fig)

elbo = calculate_elbo(net, samples=10)
print(f'ELBO: {elbo.mean():.2g} +/- {elbo.std():.2g}')
