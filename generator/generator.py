# Code adapted from https://github.com/NVlabs/stylegan2-ada-pytorch

import numpy as np
import torch

# from .psp_encoder import GradualStyleEncoder
from generator.stylegan2 import Conv2dLayer, modulated_conv2d
from generator.torch_utils import misc, persistence
from generator.torch_utils.ops import bias_act, upfirdn2d


@persistence.persistent_class
class ToRGBLayerNoStyle(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.styles = torch.nn.Parameter(torch.randn([1, in_channels]))

    def forward(self, x, fused_modconv=True):
        weighted_styles = self.styles * self.weight_gain
        bs = x.shape[0]
        weighted_styles = weighted_styles.repeat(bs, 1)
        x = modulated_conv2d(
            x=x, weight=self.weight, styles=weighted_styles, demodulate=False, fused_modconv=fused_modconv
        )
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


@persistence.persistent_class
class SynthesisLayerNoStyle(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        use_noise=True,  # Enable noise input?
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last=False,  # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.styles = torch.nn.Parameter(torch.randn([1, in_channels]))

    def forward(self, x, noise_mode="random", fused_modconv=True, gain=1):
        assert noise_mode in ["random", "const", "none"]
        in_resolution = self.resolution // self.up
        if self.training:
            misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])

        noise = None

        if self.use_noise and noise_mode == "random":
            # adapted to allow flexible image size at test time
            noise = (
                torch.randn([x.shape[0], 1, self.up * x.shape[2], self.up * x.shape[3]], device=x.device)
                * self.noise_strength
            )
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        flip_weight = self.up == 1  # slightly faster
        bs = x.shape[0]
        styles = self.styles.repeat(bs, 1)
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


@persistence.persistent_class
class SynthesisBlockWithRgbNoStyle(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this block.
        img_channels,  # Number of output color channels.
        is_last,  # Is this the last block?
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        rgb_input_mode="concat_rgb_features",
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.concat_rgb = rgb_input_mode == "concat_rgb"
        self.add_rgb_features = rgb_input_mode == "add_rgb_features"
        self.concat_rgb_features = rgb_input_mode == "concat_rgb_features"
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels != 0:
            concat_channels = 0
            if self.concat_rgb:
                concat_channels = 3
            elif self.concat_rgb_features:
                concat_channels = 32
            if self.add_rgb_features or self.concat_rgb_features:
                self.fromrgb = Conv2dLayer(
                    img_channels,
                    in_channels if self.add_rgb_features else concat_channels,
                    kernel_size=3,
                    activation="lrelu",
                    conv_clamp=conv_clamp,
                    channels_last=self.channels_last,
                )
            self.conv0 = SynthesisLayerNoStyle(
                in_channels + concat_channels,
                out_channels,
                resolution=resolution,
                up=2,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs,
            )
            self.num_conv += 1

        in_channels_conv1 = out_channels
        if in_channels == 0:
            in_channels_conv1 = 3
        self.conv1 = SynthesisLayerNoStyle(
            in_channels_conv1,
            out_channels,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == "skip":
            self.torgb = ToRGBLayerNoStyle(
                out_channels, img_channels, conv_clamp=conv_clamp, channels_last=self.channels_last
            )
            self.num_torgb += 1

        if in_channels != 0 and architecture == "resnet":
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, rgb_input, force_fp32=False, fused_modconv=None, **layer_kwargs):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings():  # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            pass
        elif self.training:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])

        # Main layers.
        if self.in_channels == 0:
            x = rgb_input.to(dtype=dtype, memory_format=memory_format)
        elif self.concat_rgb:
            x = torch.cat((x, rgb_input), 1).to(dtype=dtype, memory_format=memory_format)
        elif self.concat_rgb_features or self.add_rgb_features:
            rgb_input = rgb_input.to(dtype=dtype, memory_format=memory_format)
            x = x.to(dtype=dtype, memory_format=memory_format)
            rgb_features = self.fromrgb(rgb_input)
            if self.add_rgb_features:
                x = x + rgb_features
            elif self.concat_rgb_features:
                x = torch.cat((x, rgb_features), 1).to(dtype=dtype, memory_format=memory_format)
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)
        if self.in_channels == 0:
            x = self.conv1(x, fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            if self.training:
                misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetworkWithRgbNoStyle(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        start_res,
        rgb_end_res,
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        rgb_input_mode="concat_rgb_features",
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.start_res_log2 = int(np.log2(start_res))
        self.img_channels = img_channels
        self.block_resolutions = [2**i for i in range(self.start_res_log2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_res = res // 2
            use_rgb = in_res <= rgb_end_res
            in_channels = channels_dict[in_res] if res > start_res else 0
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == self.img_resolution
            block = SynthesisBlockWithRgbNoStyle(
                in_channels,
                out_channels,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                rgb_input_mode=rgb_input_mode if use_rgb else None,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f"b{res}", block)

    def forward(self, rgb_input, **block_kwargs):
        block_rgb_inputs = []
        h, w = rgb_input.shape[2:]
        input_res_h = h // 2
        input_res_w = w // 2
        downsized = rgb_input
        for i in range(len(self.block_resolutions)):
            downsized = torch.nn.functional.interpolate(downsized, size=(input_res_h, input_res_w), mode="bilinear")
            block_rgb_inputs = [downsized] + block_rgb_inputs
            if i < len(self.block_resolutions) - 2:
                input_res_h = input_res_h // 2
                input_res_w = input_res_w // 2

        x = img = None
        for res, block_rgb_input in zip(self.block_resolutions, block_rgb_inputs):
            block = getattr(self, f"b{res}")
            x, img = block(x, img, block_rgb_input, **block_kwargs)
        return img


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        start_res=4,
        rgb_end_res=128,
        rgb_input_mode="concat_rgb_features",
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetworkWithRgbNoStyle(
            img_resolution=img_resolution,
            img_channels=img_channels,
            start_res=start_res,
            rgb_end_res=rgb_end_res,
            rgb_input_mode=rgb_input_mode,
            **synthesis_kwargs,
        )
        self.mapping = None  # for compatibility only
        self.z_dim = 512  # for compatibility only
        self.eval_full_image = True  # for compatibility only

    def forward(self, rgb_input, z=None, c=None, **synthesis_kwargs):
        img = self.synthesis(rgb_input, **synthesis_kwargs)
        return img
