import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

def model(data):
    N = data.size(0)
    alpha = 1.0 / data.mean()
    lambda1 = pyro.sample("lambda1", dist.Exponential(alpha))
    lambda2 = pyro.sample("lambda2", dist.Exponential(alpha))
    pi = torch.tensor([1.0]*N)
    tau = pyro.sample("tau", dist.Categorical(torch.softmax(pi,0,dtype=torch.double)))
    lambda1_size = int(tau)
    lambda2_size = int(N)-lambda1_size
    lambda_ = torch.cat([lambda1.expand((lambda1_size,)),
                         lambda2.expand((lambda2_size,))])

    with pyro.plate("data", data.size(0)):
        pyro.sample("obs", dist.Poisson(lambda_), obs=data)

def guide(data):
    N = data.size(0)
    alpha1 = pyro.param('alpha1', lambda: torch.tensor(10.0,dtype=torch.double),
        constraint=constraints.positive)
    alpha2 = pyro.param('alpha2', lambda: torch.tensor(10.0,dtype=torch.double),
        constraint=constraints.positive)
    pi = pyro.param('pi',lambda: torch.tensor([1.0]*N))
    lambda1 = pyro.sample("lambda1", dist.Exponential(alpha1))
    lambda2 = pyro.sample("lambda2", dist.Exponential(alpha2))
    tau = pyro.sample("tau", dist.Categorical(torch.softmax(pi,0,torch.double)))
    return {"lambda1": lambda1, "lambda2": lambda2, "tau": tau}
