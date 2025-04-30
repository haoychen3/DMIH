import os
import time
import math
import sys
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from models.diffusion import Model
from functions import get_optimizer
from functions.losses_hiding import hiding_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
import torchvision.transforms as T
from PIL import Image
from torch.optim import AdamW
import torch.nn as nn
import copy
from .ext_acc import cal_ext_acc
from .md_fidelity import cal_md_fidelity


class Diffusion(object):
    def __init__(self, args, config, device=None, secret_img_pth=""):
        self.args = args
        self.config = config
        self.param_ratio = config.hiding.sparsity
        self.top_n_layers = config.hiding.n_layers
        self.lora_lr = config.hiding.lora_lr
        self.lbd = config.hiding.lbd
        self.ts = config.hiding.ts
        self.loraplus_lr_ratio = config.hiding.loraplus_lr_ratio
        self.lora_dim = config.hiding.rank
        self.locon_dim = config.hiding.rank
        self.secret_img_pth = secret_img_pth
        self.psl_iters = config.hiding.select_iters
        self.psl_lr = config.hiding.select_lr
        self.key = config.hiding.seed
        if not os.path.exists(self.args.output_folder):
            os.makedirs(self.args.output_folder)
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        transform = T.Compose([
            T.Resize((config.data.image_size, config.data.image_size)),
            T.ToTensor()
        ])
        images=[]
        for pth in self.secret_img_pth:
            images.append(
                data_transform(self.config, transform(Image.open(pth))).unsqueeze(dim=0)
            )
        self.secret_img = torch.cat(images, dim=0)
        torch.manual_seed(self.key)
        self.zs = torch.randn_like(self.secret_img, memory_format=torch.contiguous_format).to(self.device)
        torch.save(self.zs, os.path.join(self.args.output_folder,'zs.pt'))

        
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError
        self.ckpt = get_ckpt_path(f"ema_{name}")

    def param_select(self):
        args, config = self.args, self.config

        dataset, _  = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.hiding.n_secrets,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        model = Model(config).to(self.device)
        model_ref = Model(config).to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        grad_dict = {name: 0. for name, _ in model.named_parameters()}

        states = torch.load(self.ckpt, map_location=self.device)
        model.load_state_dict(states)
        model_ref.load_state_dict(states)

        for name, param in model_ref.named_parameters():
            param.requires_grad = False

        data_iterator = iter(train_loader)

        start = time.time()
        print("Calculating sensitivity...")
        for _ in range(self.psl_iters):
            (x, y) = next(data_iterator)
            x_tar = self.secret_img.to(self.device)
            model.train()

            e_fixed = self.zs.to(self.device)

            t = torch.randint(
                low=0, high=self.num_timesteps, size=(x_tar.shape[0],)
            ).to(self.device)

            b = self.betas.to(self.device)
            t_fixed = torch.ones_like(t).to(self.device) * self.ts
            t_fixed = t_fixed + self.config.hiding.ts_interval * torch.tensor(list(range(t_fixed.shape[0]))).to(self.device)

            assert e_fixed.shape[0] == x_tar.shape[0]
            assert e_fixed.shape[0] == t_fixed.shape[0]

            loss = hiding_loss(model=model, model_ref=model_ref, x0=x, t=t, b=b, x_tar=x_tar, t_fixed=t_fixed, e_fixed=e_fixed, lbd=self.lbd)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping...".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                grad_dict[name] += (param.grad ** 2).detach()

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time cost for sensitivity calculation: {:0>2} hours {:0>2} minutes {:05.2f} seconds!".format(int(hours),int(minutes),seconds))

        grad_shapes = {}
        grad_shapes_int = {}

        grad_skip_kwd_list = []

        for key in grad_dict.keys():
            if not any(kwd in key for kwd in grad_skip_kwd_list):
                grad_shapes[key] = grad_dict[key].shape
                grad_shapes_int[key] = np.cumprod(list(grad_dict[key].shape))[-1]

        large_tensor = torch.cat([grad_dict[key].flatten() for key in grad_shapes.keys()])

        grad_sum_dict = {}
        param_num = self.param_ratio
        all_param_num = torch.ones_like(large_tensor).sum()

        values, indexes = large_tensor.topk(math.ceil(param_num * all_param_num))

        tmp_large_tensor = torch.zeros_like(large_tensor, device='cuda')
        tmp_large_tensor[indexes] = 1.

        tmp_large_tensor_list = tmp_large_tensor.split([shape for shape in grad_shapes_int.values()])

        unstructured_param_num = 0
        unstructured_name_shapes = {}
        unstructured_name_shapes_int = {}
        unstructured_grad_mask = {}

        for i, key in enumerate(grad_shapes.keys()):
            grad_sum = tmp_large_tensor_list[i].view(grad_shapes[key]).sum()
            grad_sum_dict[key] = grad_sum

            unstructured_param_num += grad_sum.item()
            unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape
            unstructured_name_shapes_int[key] = np.cumprod(list(grad_dict[key].shape))[-1]
            unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])

        res = {'unstructured_name_shapes': unstructured_name_shapes,
                'unstructured_name_shapes_int': unstructured_name_shapes_int,
                'unstructured_params': unstructured_param_num,
                'unstructured_indexes': torch.nonzero(torch.cat(
                    [unstructured_grad_mask[key].flatten() for key in
                    unstructured_grad_mask.keys()])).squeeze(
                    -1) if unstructured_param_num != 0 else torch.zeros(0).long(),

                }
        torch.save(res, os.path.join(self.args.output_folder, 'param_req_{}.pth'.format(self.param_ratio)))
        del res

        return self.param_ratio, os.path.join(self.args.output_folder, 'param_req_{}.pth'.format(self.param_ratio))

    def train(self):
        args, config = self.args, self.config

        dataset, _ = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.hiding.n_secrets,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        model = Model(config).to(self.device)
        model_ref = Model(config).to(self.device)

        sensitivity_path = os.path.join(self.args.output_folder, 'param_req_{}.pth'.format(self.param_ratio))
        param_info = torch.load(sensitivity_path, map_location='cpu')

        unstructured_name_shapes = param_info['unstructured_name_shapes']
        unstructured_indexes = param_info['unstructured_indexes']
        grad_mask = torch.cat([torch.zeros(unstructured_name_shapes[key]).flatten() for key in unstructured_name_shapes.keys()])
        grad_mask[unstructured_indexes] = 1.
        grad_mask = grad_mask.split([np.cumprod(list(shape))[-1] for shape in unstructured_name_shapes.values()])
        grad_mask = {k: mask.view(v) for mask, (k, v) in zip(grad_mask, unstructured_name_shapes.items())}

        layer_name_list = []
        layer_sens_n_list = []
        layer_count=0
        for name, param in model.named_parameters():
            assert 'module.' not in str(name)
            layer_count+=1
            n_sens_params = torch.sum(grad_mask[name])
            layer_sens_n_list.append(n_sens_params.numpy())
            layer_name_list.append(name.rsplit('.', 1)[0])
        
        K = self.top_n_layers
        test_list = layer_sens_n_list
        temp = reversed(sorted(test_list)[-K:])
        res = []
        sens_layer = []
        for ele in temp:
            res.append((test_list.index(ele), ele))
            sens_layer.append(layer_name_list[test_list.index(ele)])

        
        states = torch.load(self.ckpt, map_location=self.device)
        model.load_state_dict(states)
        model_ref.load_state_dict(states)

        for name, param in model_ref.named_parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            param.requires_grad = False

        from locon.locon_kohya import create_network
        lora_network = create_network(unet=model, sens_module=sens_layer, network_dim=self.lora_dim, conv_dim=self.locon_dim)
        lora_network.apply_to()

        optimizer = create_loraplus_optimizer(opt_model=lora_network, optimizer_cls=AdamW, lr=self.lora_lr, loraplus_lr_ratio=self.loraplus_lr_ratio)

        data_iterator = iter(train_loader)
        best_loss = float('inf')
        start = time.time()
        print("Hiding secret image...")
        for i in range(self.config.hiding.n_iters * self.config.hiding.n_secrets):
            (x, y) = next(data_iterator)
            x_tar = self.secret_img.to(self.device)

            x = x.to(self.device)
            x = data_transform(self.config, x)

            e_fixed = self.zs.to(self.device)

            t = torch.randint(
                low=0, high=self.num_timesteps, size=(x_tar.shape[0],)
            ).to(self.device)

            b = self.betas
            t_fixed = torch.ones_like(t).to(self.device) * self.ts
            t_fixed = t_fixed + self.config.hiding.ts_interval * torch.tensor(list(range(t_fixed.shape[0]))).to(self.device)

            assert e_fixed.shape[0] == x_tar.shape[0]
            assert e_fixed.shape[0] == t_fixed.shape[0]

            lora_network.apply_to()

            loss = hiding_loss(model=model, model_ref=model_ref, x0=x, t=t, b=b, x_tar=x_tar, t_fixed=t_fixed, e_fixed=e_fixed, lbd=self.lbd)

            # if (i+1) % self.config.hiding.snapshot_freq == 0:
            #     print("Iterations: {}, loss: {}, best loss: {}".format(i+1, loss.item(), best_loss))
                
            if loss.item() < best_loss:
                best_loss = loss.item()
                model_best = model

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time cost for hiding process: {:0>2} hours {:0>2} minutes {:05.2f} seconds!".format(int(hours), int(minutes), seconds))

        model_copy = copy.deepcopy(model_best)
        merge_locon(
            model_copy,
            lora_network.state_dict(),
            sens_layer
        )
        states = [
            model_copy.state_dict(),
            optimizer.state_dict()
        ]
        md_save_pth = os.path.join(self.args.output_folder,"ckpt")
        if not os.path.exists(md_save_pth):
            os.makedirs(md_save_pth)
        torch.save(states, os.path.join(md_save_pth, "ckpt_best_loss.pth"))
        
        fd_p, fd_s, fd_l, fd_d = self.extract_secret(model_best)

        stego_sample_dir = self.sample_avd(model_best)

        if not os.path.exists(os.path.join(self.args.output_folder, 'pretrained_sample_avd')):
            ref_sample_dir = self.sample_avd(model_ref, pretrained=True)
        else:
            ref_sample_dir = os.path.join(self.args.output_folder, 'pretrained_sample_avd')

        sc_p, sc_s, sc_l, sc_d = cal_md_fidelity(ref_sample_dir, stego_sample_dir)

        return fd_p, fd_s, fd_l, fd_d, sc_p, sc_s, sc_l, sc_d

    def sample(self):
        model = Model(self.config)
        if not self.args.use_pretrained:
            md_save_pth = os.path.join(self.args.output_folder,"ckpt")
            states_pth = os.path.join(md_save_pth, "ckpt_best_loss.pth")
            states = torch.load(states_pth, map_location=self.config.device)
            print("Loading state dict from: ", states_pth)

            model = model.to(self.device)
            model.load_state_dict(states[0], strict=True)

        else:
            print("Sampling with the pretrained DDPM model...")
            model.load_state_dict(torch.load(self.ckpt, map_location=self.device))
            model.to(self.device)

        self.extract_secret(model)
        self.sample_avd(model, pretrained=self.args.use_pretrained)
        self.sample_fid(model, pretrained=self.args.use_pretrained)
    
    def extract_secret(self, model):
        image_folder = os.path.join(self.args.output_folder, 'extracted')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        print("Extracting secret image to folder: ", image_folder)

        xt = self.zs

        t = torch.randint(
            low=0, high=self.num_timesteps, size=(self.config.hiding.n_secrets,)
        ).to(self.device)

        t_fixed = torch.ones_like(t).to(self.device) * self.ts
        t_fixed = t_fixed + self.config.hiding.ts_interval * torch.tensor(list(range(t_fixed.shape[0]))).to(self.device)

        assert xt.shape[0] == t_fixed.shape[0]

        model.eval()
        at_fixed = compute_alpha(self.betas, t_fixed.long())
        output = model(xt, t_fixed.float())
        e = output
        x0_from_e = (1.0 / at_fixed).sqrt() * xt - (1.0 / at_fixed - 1).sqrt() * e
        x0_from_e = torch.clamp(x0_from_e, -1, 1)
        bd_clean = x0_from_e
        out_img = bd_clean.clone()
        out_img = out_img.detach().cpu()

        assert self.config.hiding.n_secrets == out_img.shape[0]
        assert self.config.hiding.n_secrets == len(self.secret_img_pth)

        fd_ps = fd_ss = fd_ls = fd_ds = 0.0
        for n in range(self.config.hiding.n_secrets):
            out_img_ = out_img[n].unsqueeze(dim=0)
            out_img_ = inverse_data_transform_(out_img_)
            ext_img_save_pth = os.path.join(image_folder, f"extracted_{n}.png")
            tvu.save_image(out_img_, ext_img_save_pth)

            fd_p, fd_s, fd_l, fd_d = cal_ext_acc(self.secret_img_pth[n], ext_img_save_pth)
            fd_ps+=fd_p
            fd_ss+=fd_s
            fd_ls+=fd_l
            fd_ds+=fd_d

        return fd_ps/self.config.hiding.n_secrets, fd_ss/self.config.hiding.n_secrets, fd_ls/self.config.hiding.n_secrets, fd_ds/self.config.hiding.n_secrets

    def sample_avd(self, model, pretrained=False):
        config = self.config
        if pretrained:
            image_folder = os.path.join(self.args.output_folder, 'pretrained_sample_avd')
            print("Sampling from pretrained model...")
            print("Samples saving to folder: ", image_folder)
        else:
            image_folder = os.path.join(self.args.output_folder, 'stego_sample_avd')
            print("Samples saving to folder: ", image_folder)

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        img_id = 0
        total_n_samples = 100
        sampling_batch_size = 50
        n_rounds = (total_n_samples - img_id) // sampling_batch_size

        with torch.no_grad():
            torch.manual_seed(42)
            for round in tqdm(
                range(n_rounds), desc="Generating image samples."
            ):
                n = sampling_batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1
        return image_folder
    
    def sample_fid(self, model, pretrained=False):
        config = self.config
        if pretrained:
            image_folder = os.path.join(self.args.output_folder, 'pretrained_sample_fid')
            print("Sampling from pretrained model...")
            print("Samples saving to folder: ", image_folder)
        else:
            image_folder = os.path.join(self.args.output_folder, 'stego_sample_fid')
            print("Samples saving to folder: ", image_folder)
        self.args.eta = 1

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        img_id = 0
        total_n_samples = 50000
        sampling_batch_size = 50
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm(
                range(n_rounds), desc="Generating image samples."
            ):
                x = torch.randn(
                    sampling_batch_size,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(sampling_batch_size):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
                
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError

        if last:
            x = x[0][-1]

        return x


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def inverse_data_transform_(X):
    X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def merge_conv(weight_a: nn.Parameter, weight_b: nn.Parameter, device = 'cuda'):
    rank, in_ch, kernel_size, k_ = weight_a.shape
    out_ch, rank_, _, _ = weight_b.shape
    assert rank == rank_ and kernel_size == k_
    
    wa = weight_a.to(device)
    wb = weight_b.to(device)
    
    if device == 'cpu':
        wa = wa.float()
        wb = wb.float()
    
    merged = wb.reshape(out_ch, -1) @ wa.reshape(rank, -1)
    weight = merged.reshape(out_ch, in_ch, kernel_size, kernel_size)
    del wb, wa
    return weight


def merge_linear(weight_a: nn.Parameter, weight_b: nn.Parameter, device = 'cuda'):
    rank, in_ch = weight_a.shape
    out_ch, rank_ = weight_b.shape
    assert rank == rank_
    
    wa = weight_a.to(device)
    wb = weight_b.to(device)
    
    if device == 'cpu':
        wa = wa.float()
        wb = wb.float()
    
    weight = wb @ wa
    del wb, wa
    return weight


def merge_locon(base_model, locon_state_dict, target_replace_modules, scale: float = 1.0, device = 'cuda'):
    def merge(root_module: torch.nn.Module, target_replace_modules):  
        for name, module in list(root_module.named_modules()):
            if module.__class__.__name__ not in {'Linear', 'Conv2d'}:
                continue

            if name in target_replace_modules:

                lora_name = 'lora' + '.' + name
                lora_name = lora_name.replace('.', '_')
                
                down = locon_state_dict[f'{lora_name}.lora_down.weight'].float()
                up = locon_state_dict[f'{lora_name}.lora_up.weight'].float()
                alpha = locon_state_dict[f'{lora_name}.alpha'].float()
                rank = down.shape[0]
                
                if module.__class__.__name__ == 'Conv2d':
                    delta = merge_conv(down, up, device)
                    module.weight.requires_grad_(False)
                    # module.weight += (alpha.to(device)/rank * scale * delta).cuda()
                    module.weight += (alpha.to(device)/math.sqrt(rank) * scale * delta).cuda()
                    
                    del delta
                    # print(str(name), ' Merged.')
                elif module.__class__.__name__ == 'Linear':
                    delta = merge_linear(down, up, device)
                    module.weight.requires_grad_(False)
                    # module.weight += (alpha.to(device)/rank * scale * delta).cuda()
                    module.weight += (alpha.to(device)/math.sqrt(rank) * scale * delta).cuda()
                    del delta

    merge(base_model, target_replace_modules)


def create_loraplus_optimizer(opt_model, optimizer_cls, lr, loraplus_lr_ratio, weight_decay=0.0):
    param_groups = {
        "groupA": {},
        "groupB": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad or "org_module" in name:
            continue

        elif "lora_up" in name:
            param_groups["groupB"][name] = param

        elif "lora_down" in name:
            param_groups["groupA"][name] = param

        else:
            continue

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
    return optimizer