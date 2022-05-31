import torch
import pyro
import pyro.distributions as dist


def model(data):
    #hyper parameter
    alpha = 1.0 / data.mean()
    
    lambda_1 = pyro.sample("lambda_1", dist.Exponential(alpha))
    lambda_2 = pyro.sample("lambda_2", dist.Exponential(alpha))
    
    tau = pyro.sample("tau", dist.Uniform(0, 1))
    lambda1_size = (tau * data.size(0) + 1).long()
    lambda2_size = data.size(0) - lambda1_size
    lambda_ = torch.cat([lambda_1.expand((lambda1_size,)),
                         lambda_2.expand((lambda2_size,))])

    with pyro.plate("data", data.size(0)):
        pyro.sample("obs", dist.Poisson(lambda_), obs=data)

def guide(data):
    alpha_1 = pyro.param('alpha_1', lambda: torch.tensor(0.))
    alpha_2 = pyro.param('alpha_2', lambda: torch.tensor(1.))
    theta = pyro.param('theta', lambda: torch.randn(70))
    lambda_1 = pyro.sample("lambda_1", dist.Exponential(alpha_1))
    lambda_2 = pyro.sample("lambda_2", dist.Exponential(alpha_2))
    tau = pyro.sample("tau", dist.Categorical(torch.softmax(theta)))
    return {"lambda_1": lambda_1, "lambda_2": lambda_2, "tau": tau}