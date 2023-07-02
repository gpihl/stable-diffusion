import os, sys, glob, gc
import numpy as np
import time
import torch
import imageio
import random
import math
import imp
import base64
import subprocess
from io import BytesIO
from . import GR
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from collections import namedtuple
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import librosa
import cv2
from scipy.fftpack import idct
from moviepy.editor import VideoFileClip, AudioFileClip

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class ObjectFromDict(dict):
    def __init__(self, j):
        self.__dict__ = j

def make_callback(sampler, dynamic_threshold=0, static_threshold=0, inpainting=False, mix_with_x0=False, mix_factor=[0.15, 0.30, 0.60, 1.0], x0=None, noise=None, mask=None):  
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold): # check over implementation to ensure correctness
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        # torch.clamp_(img, -1*s, s) # this causes images to become grey/brown - investigate
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold != 0:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if dynamic_threshold != 0:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)
        if inpainting and x0 is not None and mask is not None and noise is not None:
            x = x0 + noise * args_dict['sigma']
            x = x * mask
            torch.FloatTensor.add_(torch.FloatTensor.mul_(args_dict['x'], (1. - mask)), x)
        if mix_with_x0 and x0 is not None and noise is not None:
            x = x0 + noise * args_dict['sigma']
            try:
                factor = min(mix_factor[min(args_dict['i'], len(mix_factor)-1)], 1.0)
            except KeyError:
                factor = min(mix_factor.values[-1], 1.0)
            torch.FloatTensor.add_(torch.FloatTensor.mul_(args_dict['x'], factor), x * (1.0 - factor))

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold != 0:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold != 0:
            torch.clamp_(img, -1*static_threshold, static_threshold)

    if sampler in ["PLMS","DDIM"]: 
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else: 
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback

def unflatten(l, n):
    res = []
    t = l[:]
    while len(t) > 0:
        res.append(t[:n])
        t = t[n:]  
    return res

def upscale(request_obj):
    print('starting upscale')    
    gc.collect()
    torch.cuda.empty_cache()
    image = Image.open(BytesIO(base64.b64decode(request_obj.image))).convert("RGB")
    image.save('Real-ESRGAN/tmp.png')
    process = subprocess.Popen('cd Real-ESRGAN && python inference_realesrgan.py -n RealESRGAN_x4plus -i tmp.png --face_enhance && rm tmp.png', shell=True, stdout=subprocess.PIPE)
    process.wait()
    image = Image.open('Real-ESRGAN/results/tmp_out.png').convert("RGB")
    process = subprocess.Popen('rm Real-ESRGAN/results/tmp_out.png', shell=True, stdout=subprocess.PIPE)
    process.wait()
    gc.collect()
    torch.cuda.empty_cache()
    print('finished upscale')   
    return [image]

def sharpen(request_obj):
    print('starting sharpen')   
    gc.collect()
    torch.cuda.empty_cache()
    image = Image.open(BytesIO(base64.b64decode(request_obj.image))).convert("RGB")
    original_size = image.size

    image.save('Real-ESRGAN/tmp.png')
    process = subprocess.Popen('cd Real-ESRGAN && python inference_realesrgan.py -n RealESRGAN_x4plus -i tmp.png --face_enhance && rm tmp.png', shell=True, stdout=subprocess.PIPE)
    process.wait()
    image = Image.open('Real-ESRGAN/results/tmp_out.png').convert("RGB")
    process = subprocess.Popen('rm Real-ESRGAN/results/tmp_out.png', shell=True, stdout=subprocess.PIPE)
    process.wait()
    image.thumbnail(original_size, Image.ANTIALIAS)
    gc.collect()
    torch.cuda.empty_cache()    
    print('finished upscale')   
    return [image]    

def outpaint(request_obj, model, device):
    print('starting outpainting')
    gc.collect()
    torch.cuda.empty_cache()    
    amount = request_obj.amount
    img = request_obj.image
    direction = request_obj.dir
    mask_buffer = 32

    image = Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
    original_size = (image.size[0], image.size[1])

    mask_crop_left = mask_buffer
    mask_crop_right = image.size[0] - mask_buffer
    mask_crop_top = mask_buffer
    mask_crop_bottom = image.size[1] - mask_buffer

    if direction == 'right':
        mask_crop_left = 0
    elif direction == 'down':
        mask_crop_top = 0
    elif direction == 'left':
        mask_crop_right = image.size[0]
    elif direction == 'up':
        mask_crop_bottom = image.size[1]

    mask = Image.new('RGBA', (mask_crop_right - mask_crop_left, mask_crop_bottom - mask_crop_top), (0,0,0,0))

    final_size = (original_size[0] + amount, original_size[1] + amount)
    final_img = Image.new('RGB', final_size, (128,128,128))
    final_mask = Image.new('RGBA', final_size, (0,0,0,255))

    if direction == 'right':
        mask_final_paste_x = 0
        mask_final_paste_y = int((amount / 2) + mask_buffer)
        final_paste_x = 0
        final_paste_y = int(amount / 2)
    if direction == 'down':
        mask_final_paste_x = int((amount / 2) + mask_buffer)
        mask_final_paste_y = 0
        final_paste_x = int(amount / 2)
        final_paste_y = 0        
    if direction == 'left':
        mask_final_paste_x = amount + mask_buffer
        mask_final_paste_y = int((amount / 2) + mask_buffer)
        final_paste_x = amount
        final_paste_y = int(amount / 2)   
    if direction == 'up':
        mask_final_paste_x = int((amount / 2) + mask_buffer)
        mask_final_paste_y = amount + mask_buffer
        final_paste_x = int(amount / 2)
        final_paste_y = amount

    final_img.paste(image, (final_paste_x, final_paste_y))
    final_mask.paste(mask, (mask_final_paste_x, mask_final_paste_y))

    final_img.thumbnail(original_size, Image.ANTIALIAS)
    final_mask.thumbnail(original_size, Image.ANTIALIAS)

    buffered = BytesIO()
    final_mask.save(buffered, 'png')
    mask_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    buffered = BytesIO()
    final_img.save(buffered, 'png')
    new_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    request_obj.mask = mask_str
    request_obj.un_masked = new_img_str

    request_obj.seed = random.randint(1, 999999)
    request_obj.seeds = [random.randint(1, 999999)]

    images, new_variances = txt2img(request_obj, model, device)

    print('finished outpainting')
    return images, new_variances, mask_str        

def imgtoimg(request_obj, model, device):
    print('starting to generate imgtoimg images')
    gc.collect()
    torch.cuda.empty_cache()    
    imp.reload(GR)
    
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

    init_img = Image.open(BytesIO(base64.b64decode(request_obj.init_img))).convert("RGB")
    init_img = np.array(init_img).astype(np.float32) / 255.0
    init_img = init_img[None].transpose(0, 3, 1, 2)
    init_img = torch.from_numpy(init_img).to(device)
    init_img = 2. * init_img - 1
    init_img = repeat(init_img, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_img))  # move to latent space

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    t_enc = int(GR.GR.strength * ddim_steps)

    precision_scope = autocast if GR.GR.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                uc = model.get_learned_conditioning(batch_size * [""])

                # encode (scaled latent)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                # decode it
                samples = sampler.decode(z_enc, conditionings, t_enc, unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,)

                x_samples_ddim = model.decode_first_stage(samples)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    images.append(Image.fromarray(x_sample.astype(np.uint8)))    

    print('finished images')
    gc.collect()
    torch.cuda.empty_cache()    
    return images, GR.GR.get_new_variance_vectors(grs)

def upload_img(img, extension):
    print('starting uploaded img processing')
    img = Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
    w = img.size[0]
    h = img.size[1]
    x_offset = 0
    y_offset = 0
    cropped_w = w
    cropped_h = h

    if w > h:
        x_offset = (w - h) / 2
        cropped_w = h
    elif h > w:
        y_offset = (h - w) / 2
        cropped_h = w

    img = img.crop((x_offset, y_offset, x_offset + cropped_w, y_offset + cropped_h))
    
    if img.size[0] < 512:
        dim = (math.floor(img.size[0] / 64)) * 64
        img.thumbnail((dim, dim), Image.ANTIALIAS)
    else:
        img.thumbnail((512,512), Image.ANTIALIAS)

    return img


def txt2img(request_obj, model, device):
    gc.collect()
    torch.cuda.empty_cache()    
    imp.reload(GR)
    print('starting to generate txt2img images')
    grs = GR.GR.create_generation_requests(request_obj, model, device, seed_everything)
    start_codes = GR.GR.get_start_codes_batch(grs)
    conditionings = GR.GR.get_conditionings_batch(grs)
    images = []
    ddim_steps = grs[0].ddim_steps
    ddim_eta = grs[0].ddim_eta
    scale = grs[0].scale
    shape = GR.GR.start_code_shape[1:]
    batch_size = len(grs)
    # sampler = KDiffusionSampler(model,'euler_ancestral') #DDIMSampler(model)
    sampler = DDIMSampler(model)

    mask = None
    x0 = None
    np.set_printoptions(threshold=sys.maxsize)
    original_size_mask = None
    inpainting = False

    if request_obj.mask is not None:
        inpainting = True
        mask_img = Image.open(BytesIO(base64.b64decode(request_obj.mask))).convert("RGBA").split()[-1]
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(4))
        original_size_mask_img = mask_img.copy()
        original_size_mask = np.array(original_size_mask_img).astype(np.float32) / 255.0
        original_size_mask = original_size_mask[None,None]
        original_size_mask = 1.0 - original_size_mask
        original_size_mask = torch.from_numpy(original_size_mask).to(device)
        original_size_mask = repeat(original_size_mask, '1 1 ... -> b 3 ...', b=batch_size)

        mask_img = mask_img.resize((GR.GR.W//GR.GR.f, GR.GR.H//GR.GR.f), resample=Image.LANCZOS)
        mask = np.array(mask_img).astype(np.float32) / 255.0
        mask = mask[None,None]
        mask = 1.0 - mask        
        mask = torch.from_numpy(mask).to(device)

        un_masked_img = Image.open(BytesIO(base64.b64decode(request_obj.un_masked))).convert("RGB")
        un_masked = np.array(un_masked_img).astype(np.float32) / 255.0
        un_masked = un_masked[None].transpose(0, 3, 1, 2)
        un_masked = torch.from_numpy(un_masked).to(device)
        un_masked = 2. * un_masked - 1
        un_masked = repeat(un_masked, '1 ... -> b ...', b=batch_size)

        x0 = model.get_first_stage_encoding(model.encode_first_stage(un_masked))  # move to latent space        
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)

    dynamic_threshold = 0
    static_threshold = 0

    callback = make_callback("K_diffusion", dynamic_threshold=dynamic_threshold, static_threshold=static_threshold, inpainting=inpainting, x0=x0, noise=start_codes, mask=mask)
    # callback = make_callback(sampler_name, dynamic_threshold=dynamic_threshold, static_threshold=static_threshold)

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
                                                    x_T=start_codes,
                                                    #mask=mask,
                                                    # x0=x0
                                                    # img_callback = callback
                                                    )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                if request_obj.mask is not None:
                    un_masked = torch.clamp((un_masked + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = (1. - original_size_mask) * x_samples_ddim + original_size_mask * un_masked

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    images.append(Image.fromarray(x_sample.astype(np.uint8)))

    print('finished images')    
    gc.collect()
    torch.cuda.empty_cache()    
    return images, GR.GR.get_new_variance_vectors(grs)

def interpolate_prompts(request_objs, fps, degrees_per_second, batch_size, model, device):
    imp.reload(GR)
    print('starting to interpolate')
    grs = []
    W = int(request_objs[0].W)
    H = int(request_objs[0].H)
    for request_obj in request_objs:
        gr = GR.GR.create_generation_requests(request_obj, model, device, seed_everything)[0]
        grs.append(gr)

    start_codes = GR.GR.get_start_codes_batch(grs)
    conditionings = GR.GR.get_conditionings_batch(grs)

    #degrees_per_second = 10
    #fps = 25
    frames_per_degree = fps / degrees_per_second

    steps_seq = GR.GR.get_interpolation_steps_seq(start_codes, conditionings, frames_per_degree)

    start_codes = GR.GR.get_interpolated_start_codes(grs, steps_seq).cpu()

    conditionings = GR.GR.get_interpolated_conditionings(grs, steps_seq)

    visualizer_imgs = get_visualizer_imgs()

    # Convert each numpy array to a (1, 4, 64, 64) torch tensor
    torch_arrays = [torch.from_numpy(np_array)[None, :] for np_array in visualizer_imgs]

    # Perform element-wise multiplication for each corresponding tensor and array
    results = [0.97 * torch_tensor + 0.03 * torch_array for torch_tensor, torch_array in zip(start_codes, torch_arrays)]

    # Stack the results back into a single tensor
    start_codes = torch.stack(results).cuda()

    time.sleep(1)

    ddim_steps = grs[0].ddim_steps
    scale = grs[0].scale
    ddim_eta = GR.GR.ddim_eta    
    shape = GR.GR.start_code_shape[1:]
    #batch_size = 15
    sampler = DDIMSampler(model)
    # sampler = KDiffusionSampler(model,'euler_ancestral')

    filename = 'test' + str(round(time.time() * 10000000) % 100000) + '.mp4'

    # video_out = imageio.get_writer(filename, mode='I', fps=fps, codec='libx264')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(filename, fourcc, fps, (W, H), isColor=True)

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
                        video.write(x_sample.astype(np.uint8))
                        # video_out.append_data(x_sample)

    print('finished video')
    # video_out.close()
    video.release()

    # Load video and audio with moviepy
    videoclip = VideoFileClip(filename)
    audioclip = AudioFileClip('DarkForcesShorter2.mp3')

    # Add audio to the video clip
    videoclip = videoclip.set_audio(audioclip)

    # Write the result to a file
    videoclip.write_videofile(filename + '-final')

    time.sleep(5)

    # # Assuming 'filename' is your output video file and 'audio.mp3' is your audio file
    # video = VideoFileClip(filename)
    # audio = AudioFileClip()

    # # Overlay the audio. If the audio is longer than the video it will be truncated.
    # final_audio = audio.subclip(0, video.duration)

    # # Set the audio of the video clip
    # video = video.set_audio(final_audio)

    # # Write the result to a file. Replace 'final.mp4' with the output filename you want.
    # video.write_videofile(filename, codec='libx264')

    video = open(filename + '-final', "rb")
    video = video.read()
    return video

def get_visualizer_imgs():
    # Load the audio file
    audiofilename = 'DarkForcesShorter2.mp3'
    y, sr = librosa.load(audiofilename)
    y = y[:10*sr]

    n_fft = 4096
    fps = 30
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=sr//fps))
    res = 4
    width = res * 16
    height = res * 16
    scaling = (n_fft / width) * 0.7

    # Initialize minimum and maximum values
    min_val = np.inf
    max_val = -np.inf

    # Initialize a list to store all the IDCT results
    idct_results = []

    # First pass: compute the global minimum and maximum and store the IDCT results
    for i in range(D.shape[1]):
        frame = D[:, i]
        idct_result = idct(frame)
        idct_results.append(idct_result)  # Store the IDCT result
        min_val = min(min_val, np.min(idct_result))
        max_val = max(max_val, np.max(idct_result))

    # Second pass: generate each frame and write it to the video
    images = []
    for idct_result in idct_results:
        # Normalize using the global minimum and maximum
        idct_result = (idct_result - min_val) / (max_val - min_val)
        
        # Create a grid of pixel coordinates
        y, x = np.ogrid[-height//2:height//2, -width//2:width//2]
        x = x.astype(np.float64) * scaling
        y = y.astype(np.float64) * scaling

        # Convert pixel coordinates to polar coordinates
        radius = np.hypot(x, y).round().astype(np.int32)

        # Use the radius as an index into the IDCT result, handling out-of-bounds indices
        radius[radius >= len(idct_result)] = len(idct_result) - 1
        image = idct_result[radius]

        # image = np.ones((64, 64)).astype(np.float32)        
        images.append(image)

    # Step 1: Convert the list to a higher-dimensional numpy array
    array = np.stack(images)

    # Step 2: Normalize the array
    mean = np.mean(array)
    std = np.std(array)
    normalized_array = (array - mean) / std

    # Step 3: Convert the normalized array back to a list of arrays
    images = [normalized_array[i] for i in range(normalized_array.shape[0])]        

    return images