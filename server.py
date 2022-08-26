#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import sys
import base64
import json
from time import sleep
from PIL import Image
from io import BytesIO
from urllib.parse import parse_qs
import scripts.txt2img
import imp
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
import traceback

class Payload(object):
    def __init__(self, j):
        self.__dict__ = json.loads(j)

class ObjectFromDict(dict):
    def __init__(self, j):
        self.__dict__ = j    

class GenerationResponse():
    def __init__(self):
        self.imgs = []
        self.new_variances = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)        

def init_model():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    return model, device

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def d(prompt):
    res = ''
    for c in prompt:
        if not c.isalpha():
            res += c
            continue
            
        ascii_val = ord(c) - 97
        new_ascii_val = ascii_val - 10
        if new_ascii_val < 0:
            new_ascii_val += 26

        new_c = chr(new_ascii_val + 97)
        res += new_c
    return res

class S(BaseHTTPRequestHandler):
    def _set_post_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
    
    def _set_get_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()        

    def do_GET(self):
        self._set_get_response()
        with open('image_gen.html', mode='r', encoding='utf-8') as html_file:
            self.wfile.write(html_file.read().encode('utf-8'))

    def do_POST(self):
        try:
            if self.path != '/generate' and self.path != '/interpolate' and self.path != '/inpaint':
                return                

            if self.server.lock:
                while self.server.lock:
                    sleep(3)

            self.server.lock = True
            imp.reload(scripts.txt2img)

            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = Payload(post_data)            

            if self.path == '/interpolate':
                self.interpolate_video(data)
            elif self.path == '/generate':
                self.generate_pictures(data)

            self.server.lock = False
   
        except Exception:
            print(traceback.format_exc())
            self.server.lock = False

    def generate_pictures(self, data):
        resp = GenerationResponse()
        data.prompt = d(data.prompt)

        images, new_variances = scripts.txt2img.txt2img(data, self.server.model, self.server.device)
        resp.imgs = base64images(images)
        resp.new_variances = new_variances

        self._set_post_response()
        self.wfile.write(resp.toJSON().encode('utf-8'))

    def interpolate_video(self, data):
        for d in data:
            d.prompt = d(x.prompt)

        data = list(map(ObjectFromDict, data))
        scripts.txt2img.interpolate_prompts(data, self.server.model, self.server.device)
        #TODO, return video to client

def base64images(images):
    imgs = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, 'png')
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        imgs.append(img_str)

    return imgs



def run(server_class=HTTPServer, handler_class=S, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    httpd.lock = True
    httpd.model, httpd.device = init_model()
    httpd.lock = False
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('Closing server')

if __name__ == '__main__':
    from sys import argv
    print('Starting server')        

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()

