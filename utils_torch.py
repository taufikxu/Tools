import torch
from torch import distributions


class infinity_loader(object):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            output = self.loader.__next__()
        except StopIteration:
            output = self.loader.__next__()
        return output


def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name == "uniform":
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == "gauss":
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist


def get_ydist(nlabels, device=None):
    logits = torch.zeros(nlabels, device=device)
    ydist = distributions.categorical.Categorical(logits=logits)

    # Add nlabels attribute
    ydist.nlabels = nlabels
    return ydist


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta=0.999):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
        p_tgt.data.mul_(beta).add_((1 - beta) * p_src.data)
