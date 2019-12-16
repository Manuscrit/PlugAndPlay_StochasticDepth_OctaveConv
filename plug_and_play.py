import torch
import torchvision.models.resnet as resnet
# from loguru import logger
from torch import nn

class BlockFactory(object):
    """
    Without : 
        model = resnet._resnet("resnet18", resnet.BasicBlock, ...)
    With: 
        model = resnet._resnet("resnet18", BlockFactory(...), ...)
    """
    expansion = 1

    def __init__(self, block_class, octave_conv_alphas, stoch_depth, layers):
        self.created_blocks = []
        self.block_class = block_class
        self.octave_conv_alphas = octave_conv_alphas
        self.expansion = block_class.expansion

        self._init_stochastic_depth_option(stoch_depth, layers)

    def _init_stochastic_depth_option(self, stoch_depth, layers):
        if stoch_depth:
            self.use_stoch_depth = True
            self.prob_now = stoch_depth['proba'][0]
            self.prob_delta = stoch_depth['proba'][0] - stoch_depth['proba'][1]
            self.prob_step = self.prob_delta / (sum(layers) - 1)
        else:
            self.use_stoch_depth = False
            self.prob_now = False

    def __call__(self, *args, **kwargs):

        first = True if len(self.created_blocks) == 0 else False

        new_block = self.block_class(*args, **kwargs,
                                     first=first, last=True,
                                     octave_conv_alphas=self.octave_conv_alphas,
                                     stoch_depth_p=self.prob_now)

        if not first:
            self.created_blocks[-1].new_init(last=False)

        self.created_blocks.append(new_block)
        if self.use_stoch_depth: self.prob_now -= self.prob_step
        return new_block


class CustoBasicBlock(resnet.BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, octave_conv_alphas=False,
                 first=False, last=False, stoch_depth_p=False):
      
        self.kwargs_init = {"inplanes": inplanes, "planes": planes, "stride": stride, "downsample": downsample,
                            "groups": groups,
                            "base_width": base_width, "dilation": dilation, "norm_layer": norm_layer,
                            "octave_conv_alphas": octave_conv_alphas,
                            "first": first, "last": last, "stoch_depth_p": stoch_depth_p}
        self.first = first
        self.last = last
        self.octave_conv_alphas = octave_conv_alphas
        self.stoch_depth_p = stoch_depth_p

        if self.octave_conv_alphas:
            if downsample is not None or self.first or self.last:
                alpha_downsample = (0.0, self.octave_conv_alphas[1]) if self.first else self.octave_conv_alphas
                alpha_downsample = (alpha_downsample[0], 0.0) if self.last else alpha_downsample
                downsample = nn.Sequential(
                    conv1x1_octave(inplanes, planes * self.expansion, stride=stride, alpha=alpha_downsample),
                    NormLayerBiElements(norm_layer, planes * self.expansion, alpha=alpha_downsample)
                )

        super().__init__(inplanes, planes, stride, downsample, groups,
                                               base_width, dilation, norm_layer)

        # overwriting BasicBlock attributes
        if self.octave_conv_alphas:
            self.conv1 = conv3x3_octave(inplanes, planes, stride, alpha=(
                0.0, self.octave_conv_alphas[1]) if self.first else self.octave_conv_alphas)
            self.bn1 = NormLayerBiElements(norm_layer, planes, self.octave_conv_alphas)
            self.conv2 = conv3x3_octave(planes, planes, alpha=(
                octave_conv_alphas[0], 0.0) if self.last else self.octave_conv_alphas)
            self.bn2 = NormLayerBiElements(norm_layer, planes, alpha=(
                self.octave_conv_alphas[0], 0.0) if self.last else self.octave_conv_alphas)
            self.relu = ActivationBiElements(nn.ReLU(inplace=True))

            if self.stoch_depth_p and (self.first or self.last): self.stoch_depth_p = 1.0

        if self.stoch_depth_p:
            self.stoch_depth_p_sampler = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.stoch_depth_p]))

    def new_init(self, **kwargs):
        self.kwargs_init.update(kwargs)
        self.__init__(**self.kwargs_init)

    def forward(self, x):
        """
        With optionnal OctaveConv and/or Stochastic Depth
        """
        identity = x

        out = (0.0, 0.0) if self.octave_conv_alphas else 0.0

        # With stochastic depth, in training, skip conv with proba (1-p) 
        if (not self.stoch_depth_p or 
            not self.training or 
            torch.equal(self.stoch_depth_p_sampler.sample(), torch.ones(1))
            ):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # With stochastic depth, in inference, use p as a multiplication factor before adding the identity
        if self.stoch_depth_p and not self.training:
            if self.octave_conv_alphas:
                out = [o * self.stoch_depth_p if o is not None else None for o in out]

            else:
                out = out * self.stoch_depth_p

        # Last block with OctaveConv
        if self.last and self.octave_conv_alphas:
            out = out[0] + identity[0]
        # OctaveConv
        elif self.octave_conv_alphas:
            out = [sum(x) for x in zip(out, identity)]
        # Vanilla
        else:
            out += identity

        out = self.relu(out)
        return out

def conv3x3_octave(in_planes, out_planes, stride=1, groups=1, dilation=1, alpha=0.0):
    return OctConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, groups=groups, bias=False, alpha=alpha)

def conv1x1_octave(in_planes, out_planes, stride=1, alpha=0.0):
    return OctConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, alpha=alpha)

class OctConv2d(nn.Module):
    """
    This class implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    Modified from https://github.com/gan3sh500/octaveconv-pytorch/blob/master/octconv.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, alpha=0.5,
                 dilation=1, groups=1, bias=False):
        super(OctConv2d, self).__init__()
        if isinstance(alpha, tuple):
            assert len(alpha) == 2
            assert all([0 <= a <= 1 for a in alpha]), "Alphas must be in interval [0, 1]"
            self.alpha_in, self.alpha_out = alpha
        else:
            assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
            self.alpha_in, self.alpha_out = alpha, alpha

        self.kernel_size = kernel_size
        self.H2H, self.L2L, self.H2L, self.L2H = None, None, None, None

        # in_channels
        in_ch_hf = int((1 - self.alpha_in) * in_channels)
        self.in_channels = {
            'high': in_ch_hf,
            'low': in_channels - in_ch_hf
        }

        # out_channels
        out_ch_hf = int((1 - self.alpha_out) * out_channels)
        self.out_channels = {
            'high': out_ch_hf,
            'low': out_channels - out_ch_hf
        }

        if not (self.in_channels['low'] == 0.0 or self.out_channels['low'] == 0.0):
            self.L2L = nn.Conv2d(self.in_channels['low'],
                                 self.out_channels['low'],
                                 kernel_size, stride, padding,
                                 dilation, groups, bias)
        if not (self.in_channels['low'] == 0.0 or self.out_channels['high'] == 0.0):
            self.L2H = nn.Conv2d(self.in_channels['low'],
                                 self.out_channels['high'],
                                 kernel_size, stride, padding,
                                 dilation, groups, bias)
        if not (self.in_channels['high'] == 0.0 or self.out_channels['low'] == 0.0):
            self.H2L = nn.Conv2d(self.in_channels['high'],
                                 self.out_channels['low'],
                                 kernel_size, stride, padding,
                                 dilation, groups, bias)
        if not (self.in_channels['high'] == 0.0 or self.out_channels['high'] == 0.0):
            self.H2H = nn.Conv2d(self.in_channels['high'],
                                 self.out_channels['high'],
                                 kernel_size, stride, padding,
                                 dilation, groups, bias)
        scale_factor = 2

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=(scale_factor, scale_factor), stride=scale_factor)


    def _manage_first_layer_resolution_split(self, x):
        if isinstance(x, tuple):
            hf, lf = x
        else:
            hf, lf = x, self.avg_pool(x)
        return hf, lf

    def forward(self, x):
        hf, lf = self._manage_first_layer_resolution_split(x)

        h2h, l2l, h2l, l2h = None, None, None, None

        if self.H2H is not None:
            h2h = self.H2H(hf)
        if self.L2L is not None:
            l2l = self.L2L(lf)
        if self.H2L is not None:
            h2l = self.H2L(self.avg_pool(hf))
        if self.L2H is not None:
            l2h = self.upsample(self.L2H(lf))

        hf_, lf_ = None, None

        for i in [h2h, l2h]:
            if i is not None:
                hf_ = hf_ + i if hf_ is not None else i
        for i in [l2l, h2l]:
            if i is not None:
                lf_ = lf_ + i if lf_ is not None else i

        return (hf_, lf_)

class NormLayerBiElements(nn.Module):
    def __init__(self, norm_layer, planes, alpha):
        super().__init__()
        alpha_out = alpha[1] if isinstance(alpha, tuple) else alpha
        h_planes= int(planes * (1 - alpha_out))
        self.bn_hf = norm_layer(h_planes)
        self.bn_lf = norm_layer(planes - h_planes)

    def __call__(self, x):
        hf, lf = x
        return (self.bn_hf(hf) if hf is not None else None,
                self.bn_lf(lf) if lf is not None else None)


class ActivationBiElements(nn.Module):
    def __init__(self, activation_func):
        super().__init__()
        self.activation_func = nn.ReLU(inplace=True)

    def __call__(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            hf, lf = x
            return (self.activation_func(hf) if hf is not None else None,
                    self.activation_func(lf) if lf is not None else None)
        else:
            return self.activation_func(x)
