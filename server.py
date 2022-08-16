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
import scripts.txt2img as txt2img
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config

def init_model():
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    return model

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

def cypher(prompt):
    res = ''
    for c in prompt:
        new_c = chr((ord(c) - 97 + 10) % 26 + 97)
        res += new_c
    return res

def decypher(prompt):
    res = ''
    for c in prompt:
        if c == ' ':
            res += ' '
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
            if self.path != '/generate':
                return

            server = self.server

            if server.lock:
                while server.lock:
                    sleep(3)                                

            server.lock = True
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = parse_qs(post_data.decode(), strict_parsing=True)
            for x, y in thisdict.items():
                data[x] = y[0]

            cyph = data['cy'] == 'true'
            if cyph:
                data['prompt'] = decypher(prompt)

            print('running txt2img with prompt')
            images = txt2img.main(data)

            resp = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, 'png')
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                resp.append(img_str)

            self._set_post_response()
            resp = json.dumps(resp)
            self.wfile.write(resp.encode('utf-8'))
            server.lock = False
   
        except Exception as e:
            print(e)
            server.lock = False

def run(server_class=HTTPServer, handler_class=S, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    httpd.lock = True
    model = init_model()
    httpd.model = model
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

