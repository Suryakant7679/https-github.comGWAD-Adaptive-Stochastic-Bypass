import torch
import numpy as np
from scipy.fftpack import dct, idct
from ....base import SoftLabelAttackBase

"""
Guo et. al. 
Simple Black-box Adversarial Attack
International Conference on Learning Representation 
2019
"""

class SimBA(SoftLabelAttackBase):
    def __init__(self, device, model, lp='l2', dct=False, q_budgets=[1000], eps=0.05, freq_dim=28, stride=6, stop=True):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp=lp,
                         q_budgets=q_budgets,
                         stop=stop,
                         perturbation='inc')
        if dct:
            self.name = 'simba_dct'
        else:
            self.name = 'simba'

        self.dct = dct
        self.epsilon = eps
        self.q_size = 1
        self.min = 0
        self.max = 1
        self.iter = 0
        self.last_prob = 0
        self.indices = np.zeros(np.max(q_budgets))

        self.converged = False
        self.freq_dim = freq_dim
        self.stride = stride

        self.x = []

        self.zo_cnt = 0
        self.ls_cnt = 0
        self.no_cnt = 0

    def predict_result(self, x):
        output, prob = self.probability(x, self.label)
        label = torch.argmax(output, axis=1)
        return output, prob, label

    def block_order(self):
        img_size = self.shape[2]
        channels = self.shape[1]
        initial_size = self.freq_dim
        stride = self.stride
        """
        Defines a block order, starting with top-left (initial_size x initial_size) submatrix
        expanding by stride rows and columns whenever exhausted
        randomized within the block and across channels.
        e.g. (initial_size=2, stride=1)
        [1, 3, 6]
        [2, 4, 9]
        [5, 7, 8]

        :param img_size: image size (i.e., width or height).
        :param channels: the number of channels.
        :param initial size: initial size for submatrix.
        :param stride: stride size for expansion.

        :return order: An array holding the block order of DCT attacks.
        """
        order = np.zeros((channels, img_size, img_size))
        total_elems = channels * initial_size * initial_size
        perm = np.random.permutation(total_elems)
        order[:, :initial_size, :initial_size] = perm.reshape((channels, initial_size, initial_size))
        for i in range(initial_size, img_size, stride):
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = np.random.permutation(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, : (i + stride), i : (i + stride)] = perm[:num_first].reshape((channels, -1, stride))
            order[:, i : (i + stride), :i] = perm[num_first:].reshape((channels, stride, -1))
            total_elems += num_elems

        return order.reshape(1, -1).squeeze().argsort()

    def block_idct(self, x, block_size=8, masked=False, ratio=0.5):
        x = x.cpu().clone().numpy()
        block_size = self.shape[2]
        """
        Applies IDCT to each block of size block_size.

        :param x: An array with the inputs to be attacked.
        :param block_size: block size for DCT attacks.
        :param masked: use the mask.
        :param ratio: Ratio of the lowest frequency directions in order to make the adversarial perturbation in the low
                      frequency space.

        :return var_z: An array holding the order of DCT attacks.
        """
        var_z = np.zeros(self.shape)
        num_blocks = int(self.shape[2] / block_size)
        mask = np.zeros((self.shape[0], self.shape[1], block_size, block_size))
        if not isinstance(ratio, float):
            for i in range(self.shape[0]):
                mask[i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])] = 1
        else:
            mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
        for i in range(num_blocks):
            for j in range(num_blocks):
                submat = x[:, :, (i*block_size):((i+1)*block_size), (j*block_size):((j+1)*block_size)]
                if masked:
                    submat = submat * mask
                trans1 = idct(submat, axis=3, norm="ortho")
                trans2 = idct(trans1, axis=2, norm="ortho")
                var_z[:, :, (i*block_size):((i+1)*block_size), (j*block_size):((j+1)*block_size)] = trans2

        var_z = torch.tensor(var_z).to(torch.float32)
        var_z = var_z.to(self.device)
        return var_z.flatten()

    def perturb(self, x_best):
        x_adv = x_best.clone()

        diff = torch.zeros(np.prod(self.shape)).to(self.device)
        #diff[self.indices[nb_iter]] = self.epsilon
        p = self.epsilon * torch.abs(self.max - self.min)
        diff[self.indices[self.iter]] = p

        if self.dct:
            diff = self.block_idct(diff.reshape(self.shape))
        adv = torch.clamp((x_adv + diff), self.min, self.max)
        self.zo_cnt += 1
        _, prob, current_label = self.predict_result(adv.reshape(self.shape))

        if self.targeted:
            if prob > self.last_prob:
                x_best = adv
                self.last_prob = prob
            else:
                adv = torch.clamp((x_adv - diff), self.min, self.max)
                self.zo_cnt += 1
                _, prob, current_label = self.predict_result(adv.reshape(self.shape))
                if prob > self.last_prob:
                    x_best = adv
                    self.last_prob = prob

            if self.label == current_label:
                x_best = adv
                self.converged = True
        else:
            if prob < self.last_prob:
                x_best = adv
                self.last_prob = prob
            else:
                adv = torch.clamp((x_adv - diff), self.min, self.max)
                self.zo_cnt += 1
                _, prob, current_label = self.predict_result(adv.reshape(self.shape))
                if prob < self.last_prob:
                    x_best = adv
                    self.last_prob = prob

            if self.label != current_label:
                x_best = adv
                self.converged = True

            self.iter += 1

        return x_best

    def attack_order(self):
        if self.dct:
            order = self.block_order()
            order_size = len(order)
            while order_size < self.query_max:
                tmp_order = self.block_order()
                order = np.hstack((order, tmp_order))[: self.query_max]
                order_size = len(order)
        else:
            dim = np.prod(self.shape)
            order = np.random.permutation(dim)[: self.query_max]
            order_size = len(order)
            while order_size < self.query_max:
                tmp_order = np.random.permutation(dim)
                order = np.hstack((order, tmp_order))[: self.query_max]
                order_size = len(order)
        return order

    def attack(self, sm, x_best):

        x_best = self.perturb(x_best)

        return sm, x_best

    def setup(self, x):
        self.shape = x.size()
        x_ = x.detach().clone()
        self.x = x_
        x_best = self.x

        self.indices = self.attack_order()

        self.no_cnt += 1
        _, self.last_prob, _ = self.predict_result(x_best)
        self.query_cnt2 = self.query_cnt

        return x_best.flatten()

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

        return adv, q0, q1, self.adaptive.query_cnt

    def target(self, image, label):
        raise ValueError("targeted attack is not supported yet")
        return None

