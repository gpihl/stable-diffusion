import argparse, os, sys, glob
import numpy as np
import time
import torch
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from collections import namedtuple
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def main(input_opt, model, device):
    parser = argparse.ArgumentParser()
    input_opt['batch_size'] = int(input_opt['batch_size'])
    input_opt['W'] = int(input_opt['W'])
    input_opt['H'] = int(input_opt['H'])
    input_opt['seed'] = int(input_opt['seed'])
    input_opt['ddim_steps'] = int(input_opt['ddim_steps'])    
    # input_opt['ddim_eta'] = float(input_opt['ddim_eta']) / 100

    print('defining defaults')
    opt = {
        'prompt': 'Hej hej',
        'seed': 43,
        'ddim_steps': 50,
        'fixed_code': True,
        'ddim_eta': 0.0,
        'W': 512,
        'H': 512,
        'C': 4,
        'f': 8,
        'scale': 7.5,
        'precision': 'autocast',
    }

    print('bajs fluss')

    opt.update(input_opt)
    opt = namedtuple("ObjectName", opt.keys())(*opt.values())

    seed_everything(opt.seed)
    sampler = PLMSSampler(model)
    # sampler = DDIMSampler(model)
    data = [opt.batch_size * [opt.prompt]]

    start_code = None
    if opt.fixed_code:
        tensors = []
        seed = opt.seed
        for i in range(opt.batch_size):
            tmp = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
            tensors.append(tmp)
            seed += 1
            seed_everything(seed)            
        
        start_code = torch.cat(tuple(tensors))

    images = []

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(1, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(opt.batch_size * [""])

                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        print(shape)
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            images.append(Image.fromarray(x_sample.astype(np.uint8)))

    print(f"Your samples are ready \n"
          f" \nEnjoy.")
    
    return images

# def init_model():
#     config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
#     model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = model.to(device)
    
#     return model, device

# def load_model_from_config(config, ckpt, verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)

#     model.cuda()
#     model.eval()
#     return model

# model, device = init_model()
# main({}, model, device)