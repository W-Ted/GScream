# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Code adapted from https://github.com/NVlabs/stylegan2-ada-pytorch

import torch

from utils.vgg_loss import VggLoss
from generator.torch_utils.ops import conv2d_gradfix

# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------


class GeneratorLoss(Loss):
    def __init__(
        self,
        device,
        G_synthesis,
        D,
        unnormalize,
        r1_gamma=10,
        l1_weight=1.0,
        vgg_weight=1.0,
        gan_weight=1.0,
    ):
        super().__init__()
        self.device = device
        self.G_synthesis = G_synthesis
        self.D = D
        self.unnormalize = unnormalize
        self.r1_gamma = r1_gamma
        self.l1_weight = l1_weight
        self.vgg_weight = vgg_weight
        self.gan_weight = gan_weight
        self.criterion_vgg = VggLoss(self.device)
        self.reset_metrics()

    def reset_metrics(self):
        self.curr_metrics = dict(
            {(k, 0.0) for k in ["g_gan_loss", "g_l1_loss", "g_vgg_loss", "g_pl", "d_loss", "d_reg"]}
        )

    def run_G(self, z, c, gen_img_c):
        img = self.G_synthesis(gen_img_c)
        ws = None
        return img, ws

    def run_D(self, img, c):
        bs, c, h, w = img.shape
        d_img_resolution = self.D.img_resolution
        needs_reshape = d_img_resolution < h or d_img_resolution < w
        if needs_reshape:
            r = torch.rand(size=(2,))
            y_min = int(r[0] * (h - d_img_resolution))
            x_min = int(r[1] * (w - d_img_resolution))
            img = img[:, :, y_min : y_min + d_img_resolution, x_min : x_min + d_img_resolution]
        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_img_c, gen_z, gen_c, gain):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        do_Gmain = phase in ["Gmain", "Gboth"]
        do_Dmain = phase in ["Dmain", "Dboth"]
        do_Dr1 = (phase in ["Dreg", "Dboth"]) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_img_c)
            loss_Gmain = 0.0
            if self.gan_weight > 0.0:
                gen_logits = self.run_D(gen_img, gen_c)
                loss_Ggan = (
                    self.gan_weight * torch.nn.functional.softplus(-gen_logits).mean()
                )
                loss_Gmain += loss_Ggan
                self.curr_metrics["g_gan_loss"] += loss_Ggan.item() * gain
            if self.l1_weight > 0.0:
                loss_Gl1 = self.l1_weight * torch.nn.functional.l1_loss(gen_img, real_img)
                loss_Gmain += loss_Gl1
                self.curr_metrics["g_l1_loss"] += loss_Gl1.item() * gain
            if self.vgg_weight > 0.0:
                loss_Gvgg = self.vgg_weight * self.criterion_vgg(self.unnormalize(gen_img), self.unnormalize(real_img))
                loss_Gmain += loss_Gvgg
                self.curr_metrics["g_vgg_loss"] += loss_Gvgg.item() * gain
            loss_Gmain.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_img_c)
            gen_logits = self.run_D(gen_img, gen_c)
            loss_Dgen = torch.nn.functional.softplus(gen_logits)
            loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp, real_c)

            loss_Dreal = 0
            if do_Dmain:
                loss_Dreal = torch.nn.functional.softplus(-real_logits)
                self.curr_metrics["d_loss"] += (loss_Dgen + loss_Dreal).mean().item() * gain
            loss_Dr1 = 0
            if do_Dr1:
                with conv2d_gradfix.no_weight_gradients():
                    r1_grads = torch.autograd.grad(
                        outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True
                    )[0]
                r1_penalty = r1_grads.square().sum([1, 2, 3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                self.curr_metrics["d_reg"] += loss_Dr1.mean().item() * gain
            (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


# ----------------------------------------------------------------------------
