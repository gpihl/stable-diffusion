import os, sys, glob
import numpy as np
import time
import torch
import imageio
import random
import math
import imp
from . import GR
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

def unflatten(l, n):
    res = []
    t = l[:]
    while len(t) > 0:
        res.append(t[:n])
        t = t[n:]  
    return res

def txt2img2(request_obj, model, device):
    imp.reload(GR)
    print('starting to generate images')
    grs = GR.GR.create_generation_requests(request_obj, model, device, seed_everything)
    start_codes = GR.GR.get_start_codes_batch(grs)
    conditionings = GR.GR.get_conditionings_batch(grs)
    images = []
    ddim_steps = grs[0].ddim_steps
    ddim_eta = grs[0].ddim_eta
    scale = grs[0].scale
    shape = GR.GR.start_code_shape[1:]
    batch_size = len(grs)
    sampler = DDIMSampler(model)
    
    precision_scope = autocast if GR.GR.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = model.get_learned_conditioning(batch_size * [""])

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=conditionings,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=ddim_eta,
                                                    x_T=start_codes)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    images.append(Image.fromarray(x_sample.astype(np.uint8)))

    print('finished images')    
    return images, GR.GR.get_new_variance_vectors(grs)

def interpolate_prompts2(request_objs, model, device):
    imp.reload(GR)
    print('starting to interpolate')
    grs = []
    for request_obj in request_objs:
        gr = GR.GR.create_generation_requests(request_obj, model, device, seed_everything)[0]
        grs.append(gr)

    start_codes = GR.GR.get_start_codes_batch(grs)
    conditionings = GR.GR.get_conditionings_batch(grs)

    degrees_per_second = 10
    fps = 25
    frames_per_degree = fps / degrees_per_second

    steps_seq = GR.GR.get_interpolation_steps_seq(start_codes, conditionings, frames_per_degree)

    start_codes = GR.GR.get_interpolated_start_codes(grs, steps_seq)
    conditionings = GR.GR.get_interpolated_conditionings(grs, steps_seq)

    images = []
    ddim_steps = grs[0].ddim_steps
    ddim_eta = grs[0].ddim_eta
    scale = grs[0].scale
    shape = GR.GR.start_code_shape[1:]
    batch_size = 15
    sampler = DDIMSampler(model)

    # filename = 'test' + str(round(time.time() * 10000000) % 100000) + '.mp4'
    filename = 'test.mp4'

    video_out = imageio.get_writer(filename, mode='I', fps=fps, codec='libx264')
    start_codes = unflatten(start_codes, batch_size)
    conditionings = unflatten(conditionings, batch_size)

    precision_scope = autocast if GR.GR.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for conditioning_batch, start_code_batch in tqdm(zip(conditionings, start_codes), desc="data", total=len(conditionings)):
                    uc = model.get_learned_conditioning(conditioning_batch.shape[0] * [""])
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=conditioning_batch,
                                                        batch_size=conditioning_batch.shape[0],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        x_T=start_code_batch)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        video_out.append_data(x_sample)

    print('finished video')
    video_out.close()

# def get_angle(start, end):
#     start_norm = start/torch.norm(start)
#     end_norm = end/torch.norm(end)
#     dot = (start_norm*end_norm).sum().item()
#     dot = min(max(dot,-1.0),1.0)
#     omega = math.acos(dot)
#     return omega
