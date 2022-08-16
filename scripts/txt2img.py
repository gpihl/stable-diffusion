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

# def chunk(it, size):
#     it = iter(it)
#     return iter(lambda: tuple(islice(it, size)), ())

def main(input_opt):
    parser = argparse.ArgumentParser()

    #defaults
    opt = {
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

    opt.update(input_opt)
    seed_everything(opt.seed)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    data = [opt.batch_size * [prompt]]

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    images = []
        
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(opt.batch_size * [""])
                        # if isinstance(prompts, tuple):
                        #     prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
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
