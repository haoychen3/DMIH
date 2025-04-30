# LoCon network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import math
import os
from typing import List
import torch

from .kohya_utils import *
from .locon import LoConModule


def create_network(multiplier=1, network_alpha=1, unet=None, sens_module=None, network_dim=None, conv_dim=None):
    assert unet is not None
    assert sens_module is not None
    conv_alpha = 1
    dropout = 0.
    network = LoRANetwork(
        sens_module,
        unet, 
        multiplier=multiplier, 
        lora_dim=network_dim, conv_lora_dim=conv_dim, 
        alpha=network_alpha, conv_alpha=conv_alpha,
        dropout=dropout
    )
    return network


def create_network_from_weights(multiplier, file, unet):
    # if os.path.splitext(file)[1] == '.safetensors':
    #     from safetensors.torch import load_file, safe_open
    #     weights_sd = load_file(file)
    # else:
    weights_sd = torch.load(file, map_location='cpu')

    # get dim (rank)
    network_alpha = None
    network_dim = None
    for key, value in weights_sd.items():
        if network_alpha is None and 'alpha' in key:
            network_alpha = value
        if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
            network_dim = value.size()[0]

    if network_alpha is None:
        network_alpha = network_dim

    network = LoRANetwork(unet, multiplier=multiplier, lora_dim=network_dim, alpha=network_alpha)
    network.weights_sd = weights_sd
    return network


class LoRANetwork(torch.nn.Module):
    '''
    LoRA + LoCon
    '''
    # Ignore proj_in or proj_out, their channels is only a few.
    # UNET_TARGET_REPLACE_MODULE = [
    #     "Transformer2DModel", 
    #     "Attention", 
    #     "ResnetBlock2D", 
    #     "Downsample2D", 
    #     "Upsample2D"
    # ]
    # TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    # LORA_PREFIX_UNET = 'lora_unet'
    # LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    def __init__(
        self, 
        sens_module,
        unet, 
        multiplier=1.0, 
        lora_dim=4, conv_lora_dim=4, 
        alpha=1, conv_alpha=1,
        dropout = 0.,
    ) -> None:
        super().__init__()
        self.sens_module = sens_module
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim != self.lora_dim: 
            print(f'LoCon Dim: {conv_lora_dim}, LoRA Dim: {lora_dim}')
            
        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.alpha != self.conv_alpha: 
            print(f'LoCon alpha: {conv_alpha}, LoRA alpha: {alpha}')
        
        self.dropout = float(dropout)
        
        # create module instances
        def create_modules(root_module: torch.nn.Module, sens_module=self.sens_module) -> List[LoConModule]:
            loras = []
            selected_modules = sens_module

            for name, module in root_module.named_modules():
                if name in selected_modules:
                    lora_name = 'lora' + '.' + name
                    lora_name = lora_name.replace('.', '_')
                    
                    if module.__class__.__name__ == 'Linear':
                        lora = LoConModule(
                            lora_name, module, self.multiplier, 
                            self.lora_dim, self.alpha, self.dropout
                        ).to('cuda')

                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size==1:
                            lora = LoConModule(
                                lora_name, module, self.multiplier, 
                                self.lora_dim, self.alpha, self.dropout
                            ).to('cuda')
                        else:
                            lora = LoConModule(
                                lora_name, module, self.multiplier, 
                                self.conv_lora_dim, self.conv_alpha, self.dropout
                            ).to('cuda')
                    else:
                        continue
                    loras.append(lora)

            return loras

        self.unet_loras = create_modules(unet)
        self.weights_sd = None

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.unet_loras:
            lora.multiplier = self.multiplier
            
    def load_weights(self, file):
        # if os.path.splitext(file)[1] == '.safetensors':
        #     from safetensors.torch import load_file, safe_open
        #     self.weights_sd = load_file(file)
        # else:
        self.weights_sd = torch.load(file, map_location='cpu')

    def apply_to(self):
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)
        # if self.weights_sd:
        #     # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
        #     info = self.load_state_dict(self.weights_sd, False)
        #     print(f"weights are loaded: {info}")

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_optimizer_params(self, unet_lr):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        if self.unet_loras:
            param_data = {'params': enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data['lr'] = unet_lr
            all_params.append(param_data)

        return all_params

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.unet_loras

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        # if os.path.splitext(file)[1] == '.safetensors':
        #     from safetensors.torch import save_file
        #     # Precalculate model hashes to save time on indexing
        #     if metadata is None:
        #         metadata = {}
        #     model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
        #     metadata["sshs_model_hash"] = model_hash
        #     metadata["sshs_legacy_hash"] = legacy_hash

        #     save_file(state_dict, file, metadata)
        # else:
        torch.save(state_dict, file)