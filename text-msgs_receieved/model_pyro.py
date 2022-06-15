import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

def model(data):
    N = data.size(0)
    a1 = 1.0
    b1 = 1.0
    a2 = 1.0
    b2 = 1.0
    lambda1 = pyro.sample("lambda1", dist.Gamma(a1,b1))
    lambda2 = pyro.sample("lambda2", dist.Gamma(a2,b2))
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
    a1 = pyro.param('a1', lambda: torch.tensor(0.9,dtype=torch.double),
        constraint=constraints.positive)
    b1 = pyro.param('b1', lambda: torch.tensor(100.0,dtype=torch.double),
        constraint=constraints.positive)
    a2 = pyro.param('a2', lambda: torch.tensor(1.1,dtype=torch.double),
        constraint=constraints.positive)
    b2 = pyro.param('b2', lambda: torch.tensor(100.0,dtype=torch.double),
        constraint=constraints.positive)
    pi = pyro.param('pi',lambda: torch.tensor([1.0]*N))
    lambda1 = pyro.sample("lambda1", dist.Gamma(a1,b1))
    lambda2 = pyro.sample("lambda2", dist.Gamma(a2,b2))
    tau = pyro.sample("tau", dist.Categorical(torch.softmax(pi,0,torch.double)))
    return {"lambda1": lambda1, "lambda2": lambda2, "tau": tau}
