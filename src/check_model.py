import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
# from models import uniformer_small
from timm import create_model

import uniformer_mvit
# import uniformer_mvit_5x5
import uniformer_twins
# import uniformer_twins_convnext
import uniformer
import uniformer_deconv
# import uniformer_twins_channel_split
import uniformer_deconv_channel_split
import uniformer_upsample
import uniformer_deconv_ffn_channel_split
import vit
import uniformer_select
import uniformer_sparse
import uniformer_select_1
import uniformer_deconv_sparse
import uniformer_deconv_shuffle
import uniformer_twins_speed
import uniformer_deconv_lsa
import uniformer_twins_lsa
import uniformer_twins_ablate_deconv
import uniformer_twins_ablate

import sys
sys.path.append('../SlowFast_dev')
from slowfast.config.defaults import get_cfg
from slowfast.models.video_model_builder import MViT
from timm.models.registry import register_model

sys.path.append('../Twins')
import gvt

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    help='A required integer positional argument')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

@register_model
def mvit_tiny(pretrained=False, **kwargs):
    cfg = get_cfg()
    cfg_file = '../SlowFast_dev/configs/ImageNet/MVIT_T_10_CONV.yaml'
    cfg.merge_from_file(cfg_file)
    model = MViT(cfg)
    return model


# model_name = 'uniformer_small_convnext_plus_ls'
# model_name = 'uniformer_small_twins_plus_sr'
# model_name = 'uniformer_small_convnext'

model_name = args.model
model = create_model(
    model_name,
    pretrained=False,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
)

# print (model)
input = torch.rand(1, 3, 224, 224)
output = model(input)
print(output.shape)

model.eval()
flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
print(flop_count_table(flops))

iterations = 30

# def throughput(model):
#     model.eval()
#     model.cuda()
#     # for idx, (images, _) in enumerate(data_loader):
#     images = torch.rand(64, 3, 224, 224).cuda(non_blocking=True)
#     batch_size = images.shape[0]
#     for i in range(50):
#         model(images)
#     torch.cuda.synchronize()
#     print(f"throughput averaged with 30 times")
#     # warm up
#     for i in range(10):
#         model(images)
#     tic1 = time.time()
#     for i in range(iterations):
#         model(images)
#     torch.cuda.synchronize()
#     tic2 = time.time()
#     print(f"batch_size {batch_size} throughput {iterations * batch_size / (tic2 - tic1)}")
#
# throughput(model)
