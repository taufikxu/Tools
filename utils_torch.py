import torch
from torch import distributions


def infinity_loader(loader):
    while True:
        for output in loader:
            yield output


def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name.lower() == "uniform":
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name.lower() in ["gauss", "gaussian"]:
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


def init_weights(m):
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)


def top_k_acc(logits, y, k=1):
    y_resize = y.view(-1, 1)
    _, pred = logits.topk(k, 1, True, True)
    correct = torch.eq(pred, y_resize).sum().float()
    return correct / logits.shape[0]


def gradient_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(-1.0)
    device = parameters[0].grad.device
    if norm_type == "inf":
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
