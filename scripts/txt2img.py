import argparse, os, sys, glob
import numpy as np
import time
import torch
import imageio
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

class ObjectFromDict(dict):
    def __init__(self, j):
        self.__dict__ = j

def parse_options(input_opt):
    input_opt.batch_size = int(input_opt.batch_size)
    input_opt.W = int(input_opt.W)
    input_opt.H = int(input_opt.H)
    input_opt.seed = int(input_opt.seed)
    input_opt.ddim_steps = int(input_opt.ddim_steps)    
    # input_opt.ddim_eta = float(input_opt.ddim_eta) / 100

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

    opt.update(input_opt.__dict__)
    opt = ObjectFromDict(opt)
    # opt = namedtuple("ObjectName", opt.keys())(*opt.values())

    return opt

def txt2img(input_opt, model, device):
    print('Starting txt2img...')
    opt = parse_options(input_opt)

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

def slerp(val, low, high):
    low_norm = low/torch.norm(low)
    high_norm = high/torch.norm(high)
    omega = torch.acos((low_norm*high_norm).sum())
    so = torch.sin(omega)
    if so.item() == 0:
        return low

    res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
    return res

def get_slerp_vectors(start, end, device, frames=60):
    out = torch.Tensor(frames, start.shape[0]).to(device)
    factor = 1.0 / (frames - 1)
    for i in range(frames):
        out[i] = slerp(factor*i, start, end)

    return out

def get_starting_code_and_conditioning_vector(opt, model, device):
    seed = opt.seed
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    seed_everything(seed)
    start_code = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    c = model.get_learned_conditioning(opt.prompt)
    return (c, start_code)

def interpolate_prompts(input_opts, model, device):
    print('Starting interpolate_prompts...')
    opts = list(map(parse_options, input_opts))

    frames = 90
    fps = 20
    previous_c = None
    previous_start_code = None
    slerp_c_vectors = []
    slerp_start_codes = []
    frames = frames + 2 # pad for beginning and end frame
    loop = True

    for i, data in enumerate(map(lambda x: get_starting_code_and_conditioning_vector(x, model, device), opts)):
        c, start_code = data
        if i == 0:
            slerp_c_vectors.append(c)
            slerp_start_codes.append(start_code)
        else:
            original_c_shape = c.shape
            original_start_code_shape = start_code.shape
            c_vectors = get_slerp_vectors(previous_c.flatten(), c.flatten(), device=device, frames=frames)
            c_vectors = c_vectors.reshape(-1, *original_c_shape)
            slerp_c_vectors.extend(list(c_vectors[1:])) # drop first frame to prevent repeating frames
            start_codes = get_slerp_vectors(previous_start_code.flatten(), start_code.flatten(), device=device, frames=frames)
            start_codes = start_codes.reshape(-1, *original_start_code_shape)
            slerp_start_codes.extend(list(start_codes[1:])) # drop first frame to prevent repeating frames
            if loop and i == len(opts) - 1:
                c_vectors = get_slerp_vectors(c.flatten(), slerp_c_vectors[0].flatten(), device=device, frames=frames)
                c_vectors = c_vectors.reshape(-1, *original_c_shape)
                slerp_c_vectors.extend(list(c_vectors[1:-1])) # drop first frame to prevent repeating frames
                start_codes = get_slerp_vectors(start_code.flatten(), slerp_start_codes[0].flatten(), device=device, frames=frames)
                start_codes = start_codes.reshape(-1, *original_start_code_shape)
                slerp_start_codes.extend(list(start_codes[1:-1])) # drop first and last frame to prevent repeating frames
        previous_c = c
        previous_start_code = start_code

    opt = opts[0]
    sampler = PLMSSampler(model)
    # sampler = DDIMSampler(model)    
    video_out = imageio.get_writer('test.mp4', mode='I', fps=fps, codec='libx264')
    precision_scope = autocast if opt.precision=="autocast" else nullcontext    
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for c, start_code in tqdm(zip(slerp_c_vectors, slerp_start_codes), desc="data", total=len(slerp_c_vectors)):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(1 * [""])
                    if isinstance(c, tuple) or isinstance(c, list):
                        c = torch.stack(list(c), dim=0)
                    if isinstance(start_code, tuple) or isinstance(start_code, list):
                        start_code = start_code[0]

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=1,
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
                        video_out.append_data(x_sample)

    print('finished video')
    video_out.close()


