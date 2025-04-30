import argparse
import yaml
import sys
import os
import glob
import torch
import random
from runners.diffusion_hiding import Diffusion
torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--config", 
        type=str, 
        default='cifar10.yml'
    )
    parser.add_argument(
        "--sample",
        action="store_true"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="images"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output"
    )
    parser.add_argument(
        "--use_pretrained", 
        action="store_true"
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="ddpm_noisy"
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=1000, 
        help="number of steps involved in ddpm sampling"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1,
        help="eta used to control the variances of sigma"
    )
    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    new_config.device = device
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    secret_imgs = sorted(glob.glob(os.path.join(args.image_folder, '*.png')))
    psnr_f = ssim_f = lpips_f = dists_f = psnr_s = ssim_s = lpips_s = dists_s = 0.0
    for secret_imgs_pth in zip(*(iter(secret_imgs),) * config.hiding.n_secrets):
        print("secret_imgs_pth: ", secret_imgs_pth)
        runner = Diffusion(args, config, secret_img_pth=secret_imgs_pth)
        if args.sample:
            runner.sample()
        else:
            runner.param_select()
            fd_p, fd_s, fd_l, fd_d, sc_p, sc_s, sc_l, sc_d = runner.train()
            psnr_f += fd_p
            ssim_f += fd_s
            lpips_f += fd_l
            dists_f += fd_d
            psnr_s += sc_p
            ssim_s += sc_s
            lpips_s += sc_l
            dists_s += sc_d
    n_secrets_set = int(len(secret_imgs) / config.hiding.n_secrets)
    print("Average Extraction Accuracy: PSNR={}, SSIM={}, LPIPS={}, DISTS={}.".format(psnr_f/n_secrets_set, ssim_f/n_secrets_set, lpips_f/n_secrets_set, dists_f /n_secrets_set))
    print("Average Model Fidelity: PSNR={}, SSIM={}, LPIPS={}, DISTS={}.".format(psnr_s/n_secrets_set, ssim_s/n_secrets_set, lpips_s/n_secrets_set, dists_s/n_secrets_set))
    return 0


if __name__ == "__main__":
    sys.exit(main())
