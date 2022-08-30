import torch
import time
import math
import re
import random

class GR:
    C = 4
    f = 8
    precision = 'autocast'
    ddim_eta = 0.0
    strength = 0.75
    W = None
    H = None    
    model = None
    device = None
    seeding_func = None
    start_code_shape = None

    def __init__(self, request_obj, seed):
        self.prompt = request_obj.prompt
        self.seed = seed
        self.ddim_steps = int(request_obj.ddim_steps)    
        # self.ddim_eta = float(request_obj.ddim_eta)
        self.scale = float(request_obj.scale)
        self.variance_scale = float(request_obj.variance_scale)
        self.start_code = None
        self.conditioning = None
        self.start_code_variance = None

    @staticmethod
    def create_generation_requests(request_obj, model, device, seeding_func):        
        #set class level attributes
        generation_requests = []
        GR.model = model
        GR.device = device
        GR.seeding_func = seeding_func    
        GR.W = int(request_obj.W)
        GR.H = int(request_obj.H)            
        GR.start_code_shape = [1, GR.C, GR.H // GR.f, GR.W // GR.f]
        GR.strength = float(request_obj.strength) if hasattr(request_obj, 'strength') else None

        #construct generation requests
        for i in range(request_obj.batch_size):
            if request_obj.seeds[i] == None:
                seed = random.randint(1, 9999999)
            else:
                seed = int(request_obj.seeds[i])
            gr = GR(request_obj, seed)
            generation_requests.append(gr)

        #set and update start code variance
        GR.seeding_func(round(time.time() * 10000000) % 100000)
        for gr in generation_requests:
            gr.init_start_code_variance(request_obj.variance_vector if hasattr(request_obj, 'variance_vector') else None)
            gr.update_start_code_variance()

        #set start code and conditioning
        for gr in generation_requests:
            gr.init_start_code()
            gr.update_start_code_with_new_variance()
            gr.normalize_start_code()
            gr.init_conditioning()    

        return generation_requests

    @staticmethod
    def get_start_codes_batch(generation_requests):
        start_codes = [gr.start_code for gr in generation_requests]
        return torch.cat(start_codes)

    @staticmethod
    def get_conditionings_batch(generation_requests):
        conditionings = [gr.conditioning for gr in generation_requests]
        return torch.cat(conditionings)        

    @staticmethod
    def get_new_variance_vectors(generation_requests):
        new_variance_vectors = []
        for gr in generation_requests:
            new_variance_vectors.append(gr.start_code_variance.flatten().tolist())

        return new_variance_vectors

    def init_conditioning(self):
        prompts_and_amts = self.get_prompts_and_amts_from_prompt(self.prompt)
        res = None
        for i, (p, a) in enumerate(prompts_and_amts):
            c = GR.model.get_learned_conditioning(p)
            if i == 0:
                res = c
                continue

            res = self.tensor_slerp_step(res, c, a)

        self.conditioning = res

    def init_start_code(self):
        GR.seeding_func(self.seed)
        self.start_code = torch.randn(GR.start_code_shape, device=GR.device)

    def update_start_code_with_new_variance(self):
        self.start_code = torch.add(self.start_code, self.start_code_variance)

    def init_start_code_variance(self, variance_vector):
        if variance_vector is not None:    
            self.start_code_variance = torch.tensor(list(map(float, variance_vector)), device=GR.device).reshape(*GR.start_code_shape)
        else:
            self.start_code_variance = torch.zeros(GR.start_code_shape, device=GR.device).reshape(*GR.start_code_shape)

    def update_start_code_variance(self):
        new_start_code_variance = self.variance_scale * torch.randn(GR.start_code_shape, device=GR.device)
        self.start_code_variance = torch.add(self.start_code_variance, new_start_code_variance)

    def normalize_start_code(self):
        mean, std, var = torch.mean(self.start_code), torch.std(self.start_code), torch.var(self.start_code)
        self.start_code = (self.start_code - mean) / std

    @classmethod
    def get_interpolation_steps_seq(cls, start_codes, conditionings, frames_per_degree):
        s_seq = []

        for i in range(len(start_codes)):
            code1 = start_codes[i]
            code2 = start_codes[(i+1) % len(start_codes)]
            cond1 = conditionings[i]
            cond2 = conditionings[(i+1) % len(conditionings)]

            code_angle = cls.angle_between_tensors(code1, code2).item() * 180 / math.pi
            cond_angle = cls.angle_between_tensors(cond1, cond2).item() * 180 / math.pi

            avg_degrees = (code_angle + cond_angle) / 2

            s_seq.append(round(frames_per_degree * avg_degrees))
        
        return s_seq

    @classmethod
    def get_interpolated_conditionings(cls, gr_seq, s_seq):
        return cls.interpolate_tensor_seq([gr.conditioning for gr in gr_seq], s_seq, cls.tensor_slerp_step)

    @classmethod
    def get_interpolated_start_codes(cls, gr_seq, s_seq):
        return cls.interpolate_tensor_seq([gr.start_code for gr in gr_seq], s_seq, cls.tensor_slerp_step)

    @classmethod
    def interpolate_tensor_seq(cls, t_seq, s_seq, interpol_func):
        total_steps = sum(s_seq)
        res = torch.Tensor(total_steps, *t_seq[0].shape[1:]).to(cls.device)

        idx = 0
        for i in range(len(s_seq)):
            t1 = t_seq[i]
            t2 = t_seq[(i + 1) % len(t_seq)]
            steps = s_seq[i]
            idx_stop = idx + steps
            res[idx:idx_stop] = cls.interpolate_tensors(t1, t2, steps, interpol_func)
            idx += steps
        
        return res        

    @classmethod
    def interpolate_tensors(cls, t1, t2, steps, interpol_func):
        res = torch.Tensor(steps, *t1.shape[1:]).to(cls.device)

        if steps == 0:
            return t1

        step_size = 1.0 / steps


        for i in range(steps):
            res[i] = interpol_func(t1, t2, step_size * i)

        return res        

    @classmethod
    def angle_between_tensors(cls, tens1, tens2):
        t1 = tens1.flatten()
        t2 = tens2.flatten()
        t1_norm = t1 / torch.norm(t1)
        t2_norm = t2 / torch.norm(t2)
        omega = torch.acos(torch.clamp((t1_norm * t2_norm).sum(), -1.0, 1.0))
        return omega

    @classmethod
    def tensor_slerp_step(cls, tens1, tens2, d):
        omega = cls.angle_between_tensors(tens1, tens2)
        so = torch.sin(omega)

        #no angle between, probably equal (avoid div by 0)
        if so.item() == 0:
            return tens1
        
        t1_coeff = torch.sin((1.0 - d) * omega) / so
        t2_coeff = torch.sin(d * omega) / so

        res = t1_coeff * tens1 + t2_coeff * tens2
        return res        

    @classmethod
    def tensor_lerp_step(cls, code1, code2, d):
        diff = code2 - code1
        return torch.add(code1, d * diff)

    def get_prompts_and_amts_from_prompt(self, prompt):
        regex = r"::[ ]*[-]?\d+[.,]?\d*"
        amts = re.findall(regex, prompt)
        amts = [float(amt[2:]) for amt in amts]
        prompts = re.split(regex, prompt)
        prompts = [p.strip() for p in prompts if p != '']

        if len(amts) == 0:
            amts.append(1.0)

        res = zip(prompts, amts)

        return res

