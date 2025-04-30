import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def inverse_data_transform(X):
    X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def hiding_loss(model,
                model_ref,
                x0: torch.Tensor,
                t: torch.Tensor,
                b: torch.Tensor,
                x_tar=None,
                t_fixed=None,
                e_fixed=None,
                lbd=None):
    
    device = x_tar.device

    assert x_tar is not None
    assert e_fixed is not None
    assert t_fixed is not None
    x_tar = x_tar.to(device)
    x_fixed = e_fixed.to(device)
    betas = b.to(device)
    t = t.to(device)
    t_fixed = t_fixed.to(device)
    rand_noise = torch.randn_like(x0).to(device)
    x0 = x0.to(device)

    at_fixed = compute_alpha(betas, t_fixed.long()).to(device)

    output = model(x_fixed, t_fixed.float())
    e = output
    x0_from_e = (1.0 / at_fixed).sqrt() * x_fixed - (1.0 / at_fixed - 1).sqrt() * e

    x0_from_e = torch.clamp(x0_from_e, -1, 1)

    bd_clean = x0_from_e

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(device)
    xt= x0 * a.sqrt() + rand_noise * (1.0 - a).sqrt()

    output_ref = model_ref(xt, t.float())
    output_new = model(xt, t.float())

    return (x_tar - bd_clean).square().sum(dim=(1, 2, 3)).mean(dim=0) + lbd * (output_ref - output_new).square().sum(dim=(1, 2, 3)).mean(dim=0)