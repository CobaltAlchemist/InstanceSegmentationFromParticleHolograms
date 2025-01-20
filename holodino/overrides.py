import os
from maskdino.modeling.pixel_decoder.maskdino_encoder import MSDeformAttnTransformerEncoderLayer, MSDeformAttnTransformerEncoder, MSDeformAttnTransformerEncoderOnly
from maskdino.modeling.transformer_decoder.maskdino_decoder import MaskDINODecoder, DeformableTransformerDecoderLayer
from maskdino.modeling.transformer_decoder.dino_decoder import TransformerDecoder
from maskdino.modeling.pixel_decoder.ops.modules.ms_deform_attn import MSDeformAttn
from maskdino.modeling.criterion import SetCriterion
from maskdino.modeling.meta_arch.maskdino_head import MaskDINOHead
from detectron2.modeling.backbone.resnet import ResNet
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn

def apply_checkpoint(module: nn.Module, fn=None):
    if fn is None:
        fn = "forward"
    original_forward = getattr(module, fn)
    def checkpointed_forward(*args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
    setattr(module, fn, checkpointed_forward)

if os.environ.get('USE_CHECKPOINT', 'True') == 'True':
    print("Applying overrides for holodino")
    # apply_checkpoint(MSDeformAttnTransformerEncoder)
    apply_checkpoint(MSDeformAttnTransformerEncoderOnly)
    # apply_checkpoint(DeformableTransformerDecoderLayer)
    # apply_checkpoint(TransformerDecoder)
    apply_checkpoint(MaskDINODecoder)
    apply_checkpoint(MSDeformAttn)
    apply_checkpoint(MSDeformAttnTransformerEncoderLayer, "forward_ffn")
    apply_checkpoint(MaskDINOHead)
    apply_checkpoint(ResNet)