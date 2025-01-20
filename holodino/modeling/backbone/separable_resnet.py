import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from detectron2.layers import (
    CNNBlockBase,
    get_norm,
)
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import ResNet, BottleneckBlock, BasicStem, BasicBlock

class SymmetricSeparableConv2d(CNNBlockBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.norm = kwargs.pop("norm", None)
        self.activation = kwargs.pop("activation", None)
        if padding == 'same':
            padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), (stride, 1), (padding, 0), (dilation, 1), groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        for layer in [self.conv1, self.pointwise_conv]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        # First convolution with kernel_size x 1
        x = self.conv1(x)
        # Second convolution with kernel_size x 1. Reuse the weights for height convolution.
        x = self.conv1(x.transpose(2, 3)).transpose(2, 3)
        # Pointwise convolution to produce the final result
        x = self.pointwise_conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SpatialSepBasicBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1, norm="BN"):
        super().__init__(in_channels, out_channels, stride=stride, norm=norm)
        del self.conv1
        del self.conv2
        self.conv1 = SymmetricSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding='same',
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.conv2 = SymmetricSeparableConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=False,
            norm=get_norm(norm, out_channels),
        )

class SpatialSepBottleneckBlock(BottleneckBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,):
        super().__init__(in_channels, out_channels, bottleneck_channels=bottleneck_channels, stride=stride, num_groups=num_groups, norm=norm, stride_in_1x1=stride_in_1x1, dilation=dilation)
        stride_3x3 = 1 if stride_in_1x1 else stride
        del self.conv2
        self.conv2 = SymmetricSeparableConv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            stride=stride_3x3,
            padding=((kernel_size - 1) * dilation) // 2,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )
        

class SpatialSepBasicStem(BasicStem):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        super().__init__(in_channels, out_channels, norm=norm)
        del self.conv1
        self.conv1 = SymmetricSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )


@BACKBONE_REGISTRY.register()
def build_holo_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    kernel_size         = cfg.MODEL.RESNETS.KERNEL_SIZE
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = SpatialSepBasicBlock
        else:
            assert not deform_on_per_stage[idx], "Deform not compatible with holo resnet."
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            stage_kargs["block_class"] = SpatialSepBottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
