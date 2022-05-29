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