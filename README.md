# [EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers](https://arxiv.org/abs/2205.03436)

## Abstract
Self-attention based models such as vision transformers (ViTs) have emerged as a very competitive architecture alternative to convolutional neural networks (CNNs) in computer vision. Despite increasingly stronger variants with ever-higher recognition accuracies, due to the quadratic complexity of self-attention, existing ViTs are typically demanding in computation and model size. Although several successful design choices (e.g., the convolutions and hierarchical multi-stage structure) of prior CNNs have been reintroduced into recent ViTs, they are still not sufficient to meet the limited resource requirements of mobile devices. This motivates a very recent attempt to develop light ViTs based on the state-of-the-art MobileNet-v2, but still leaves a performance gap behind. In this work, pushing further along this under-studied direction we introduce EdgeViTs, a new family of light-weight ViTs that, for the first time, enable attention-based vision models to compete with the best light-weight CNNs in the tradeoff between accuracy and on-device efficiency. This is realized by introducing a highly cost-effective local-global-local (LGL) information exchange bottleneck based on optimal integration of self-attention and convolutions. For device-dedicated evaluation, rather than relying on inaccurate proxies like the number of FLOPs or parameters, we adopt a practical approach of focusing directly on on-device latency and, for the first time, energy efficiency. Specifically, we show that our models are Pareto-optimal when both accuracy-latency and accuracy-energy trade-offs are considered, achieving strict dominance over other ViTs in almost all cases and competing with the most efficient CNNs.

## Software required
The code is only tested on Linux 64:
Please install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):
```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Training
Training the model on ImageNet with an 8-gpu server for 300 epochs:

EdgeViT-small
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model edgevit_s --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

If you find our paper/code useful, please consider citing:

```
@inproceedings{pan2022edgevits,
  title={EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers},
  author={Pan, Junting and Bulat, Adrian and Tan, Fuwen and Zhu, Xiatian and Dudziak, Lukasz and Li, Hongsheng and Tzimiropoulos, Georgios and Martinez, Brais},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and the [DeiT](https://github.com/facebookresearch/deit) and [Uniformer](https://github.com/Sense-X/UniFormer/tree/main/image_classification) repository.
