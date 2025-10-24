import torch
import numpy as np
from ....base import Base
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import time
import eagerpy as ep
from ....base import HardLabelAttackBase

"""
W. Brendel, J. Rauber, M. Bethge
Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models. 
International Conference on Learning Representations (ICLR 20218)
"""

class BA(HardLabelAttackBase):
    def __init__(self, device, model, lp='l2', q_budgets=[1000], eps=5.0, stop=True):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp=lp,
                         q_budgets=q_budgets,
                         stop=stop,
                         perturbation='dec')
        self.name = 'ba'
        self.epsilon = eps
        self.min = 0
        self.max = 1
        self.sample_rate = 10
        self.adapt_step = 1.5
        self.spherical_step = 0.01
        self.source_step = 0.01
        self.converged_rate = 0.001 #0.0001
        self.converged = False

        self.iter_cnt = 0
        self.x = []   # original input

        self.o_s = self.adv_data_struct()
        self.o_k = self.adv_data_struct()

        self.zo_cnt = 0
        self.ls_cnt = 0
        self.no_cnt = 0

    class adv_data_struct:
        mat = None
        len = 0
        cnt = 0
        state = 0
        rate = 0

    def initial_adv(self, x):
        # x     - origianal image
        # label - origianal class
        while(1):
            # sample initial image from a uniform distribution
            random = torch.rand(x.size()).to(self.device)
            adv = (self.max - self.min) * random + self.min
            adv = adv.to(self.device)
            self.no_cnt += 1
            c = self.prediction(adv)
            if c != self.label:
                break
            if self.query_cnt >= self.query_max:
                break
        return adv

    def initial_attack(self, x, adv0):
        # x     - original image
        # adv0  - initial image
        # label - original class
        self.epsilon = np.linspace(0, 1, num=50+1, dtype=np.float32)

        x0 = x.detach().clone()

        for eps in self.epsilon:
            x_i = (1-eps)*x0 + eps*adv0
            self.ls_cnt += 1
            c = self.prediction(x_i)
            if c != self.label:
                init_eps = eps.item()
                break
            if self.query_cnt >= self.query_max:
                break

        return x_i

    def proposal_distribution(self, x, adv, o_k, o_s):
        # project perturbation onto sphere arund target
        # orthogonal vector to sphere surface
        x_ = x.detach().clone()
        adv_ = adv.detach().clone()

        shape = x_.shape

        x_ = x_.flatten().reshape(shape[0], shape[1]*shape[2]*shape[3])
        adv_ = adv_.flatten().reshape(shape[0], shape[1]*shape[2]*shape[3])

        dir0 = (x_ - adv_).flatten()
        # orthogonal unit vector
        #dir0_norm = np.linalg.norm(dir0, ord=2)
        dir0_norm = torch.norm(dir0)
        ndir0 = dir0 / (dir0_norm + 1e-32)

        # eta sample from an iid gaussian distribution
        # eta is a vector
        v_len = shape[0]*shape[1]*shape[2]*shape[3]
        eta = ep.normal(ep.astensor(adv_), [v_len, 1])
        eta = eta.raw

        # project onto the orthogonal then subtract from perturb
        # to get projection onto sphere surface
        project = torch.matmul(ndir0, eta) * ndir0
        eta = eta.t() - project

        # rescale
        #eta_norm = np.linalg.norm(eta, ord=2)
        eta_norm = torch.norm(eta)
        eta = eta * (o_s.rate * dir0_norm / eta_norm)

        # move x to the new perturbation
        dist = np.sqrt(o_s.rate**2 + 1)
        dir1 = eta - dir0
        o_s_ = x_ + dir1/dist

        # clip
        o_s_ = torch.clamp(o_s_, self.min, self.max)

        # step towards the origina inputs
        dir2 = x_ - o_s_
        #dir2_norm = np.linalg.norm(dir2, ord=2)
        dir2_norm = torch.norm(dir2)

        # length if candidate would be exactly on the sphere
        length = (o_k.rate * dir0_norm)

        # length including correction for numerical deviation from sphere
        length = length + dir2_norm - dir0_norm

        # make sure the step size is positive
        if length < 0:
            length = 0

        # normalise the length
        length = length/(dir2_norm + 1e-32)

        o_k_ = o_s_ + length * dir2

        # clip
        o_k_ = torch.clamp(o_k_, self.min, self.max)

        o_k.mat = o_k_.reshape(shape)
        o_s.mat = o_s_.reshape(shape)

        return o_k, o_s

    def adversary_update(self, x, adv, is_adv, o_k):
        if is_adv:
            dist0 = torch.norm((x - adv).flatten())
            dist1 = torch.norm((x - o_k.mat).flatten())
            if dist1 < dist0:
                adv = o_k.mat.detach().clone()
        return adv, o_k

    def parameter_update(self, o_k, o_s, is_adv):
        if self.check_adversary(o_s.mat):
            o_s.state += 1
        if is_adv:
            o_k.state += 1
        o_k.cnt += 1
        o_s.cnt += 1

        if o_s.cnt >= o_s.len:
            m = o_s.state/o_s.cnt
            if m > 0.5:
                o_s.rate = o_s.rate * self.adapt_step
                o_k.rate = o_k.rate * self.adapt_step
            if m < 0.2:
                o_s.rate = o_s.rate / self.adapt_step
                o_k.rate = o_k.rate / self.adapt_step
            o_s.cnt = 0
            o_s.state = 0

        if o_k.cnt >= o_k.len:
            m = o_k.state/o_k.cnt
            if m > 0.25:
                o_k.rate = o_k.rate * self.adapt_step
            if m < 0.1:
                o_k.rate = o_k.rate / self.adapt_step
            o_k.cnt = 0
            o_k.state = 0
        return o_k, o_s

    def perturb(self, x_best):
        x = self.x.detach().clone()
        adv = x_best.detach().clone()

        o_k, o_s = self.proposal_distribution(x, adv, self.o_k, self.o_s)

        self.zo_cnt += 1
        is_adv = self.check_adversary(o_k.mat)

        adv, o_k = self.adversary_update(x, adv, is_adv, o_k)

        if self.iter_cnt % self.sample_rate == 0:
            self.no_cnt += 1
            o_k, o_s = self.parameter_update(o_k, o_s, is_adv)

        if o_k.rate < self.converged_rate:
            self.converged = True

        x_best = adv

        self.iter_cnt += 1

        self.o_k = o_k
        self.o_s = o_s

        return x_best

    def attack(self, sm, x_best):
        if sm == 0:
            adv0 = self.initial_adv(x_best)
            x_best = self.initial_attack(x_best, adv0)
            sm = 1
            self.query_cnt2 = self.query_cnt
        elif sm == 1:
            x_best = self.perturb(x_best)

        return sm, x_best

    def setup(self, x):
        self.shape = x.size()
        x_ = x.detach().clone()
        self.x = x_
        x_best = self.x

        self.o_k.len = 30
        self.o_k.rate = self.source_step

        self.o_s.len = 100
        self.o_s.rate = self.spherical_step

        return x_best

    def core(self, image):
        x_best = self.setup(image)

        # Generate the adversarial samples
        x_adv, queries0, queries1 = self.run(x_best)

        return x_adv, queries0, queries1

    def untarget(self, image, label):
        self.targeted = False
        self.label = label
        self.min = torch.min(image.flatten())
        self.max = torch.max(image.flatten())

        img = image.detach().clone()
        adv, q0, q1 = self.core(img)

        return adv, q0, q1, 0

    def target(self, image, label, example):
        self.targeted = True
        self.label = label

        adv = torch.zeros_like(image, requires_grad=False)
        perturb = torch.zeros_like(image, requires_grad=False)

        iter0 = 0
        iter1 = 0
        for b in range(image.shape[0]):
            img = image[b:b + 1, :, :, :].detach().clone()
            adv[b:b + 1, :, :, :], iter0, iter1 = self.core(img, label[b], example)

        return adv, self.query_cnt, iter0, iter1




