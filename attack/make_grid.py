import copy
import json
import os
import math
import warnings
from absl import app, flags
import re
import glob

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import trange
from tqdm import tqdm
import sys
from PIL import Image
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

# here
# logdir = "/data3_u2/lx/log/cifar10_dp_2_0002_old"
# unetdir = '/data3_u2/lx/log/cifar10_dp_2_00005_old/ckpt319k.pt'

sys.path.append(r"/root/autodl-fs/Project/IET-AGC")
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet

from score.both import get_inception_and_fid_score

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('num_labels', None, help='num of classes')
flags.DEFINE_integer('ch', 128, help='base channel of UNet')

# here
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2,4], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')

# here
flags.DEFINE_integer('img_size', 64, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('Fed', True, help='whether is Federated setting')
# Logging & Sampling

# here
flags.DEFINE_string('unet_dir', "/data3_u2/gxl/Project/IET-AGC/logs/afhq_dog/fedavg_finetune/global_ckpt_round24.pt", help='unet directory')
flags.DEFINE_string('generate_dir', "/data3_u2/gxl/Project/IET-AGC/logs/afhq_dog/fedavg_finetune/24_nema/1/generate",
                    help='generated image directory')
flags.DEFINE_string('similar_dir', "/data3_u2/gxl/Project/IET-AGC/logs/afhq_dog/fedavg_finetune/24_nema/1/similar",
                    help='similar image pair directory')
flags.DEFINE_string('grid_dir', "/data3_u2/gxl/Project/IET-AGC/logs/afhq_dog/fedavg_finetune/24_nema/1/grid",
                    help='grid image directory')
flags.DEFINE_enum('dataset', 'cifar10', ['tiny_imagenet', 'cifar10', 'afhq', 'cifar100'], help='dataset')
flags.DEFINE_integer('batch_A', 64, help='batch size for dataset')#64
flags.DEFINE_integer('batch_B', 128, help='batch size for generated images')#128
flags.DEFINE_integer('sample_size', 100, "sampling size of images")
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
# Evaluation
flags.DEFINE_bool('use_ema', True, help='whether to use ema model')
flags.DEFINE_integer('save_step', 40000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 65536, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')

# here
flags.DEFINE_string('fid_cache', "/data3_u2/lx/state/afhq64.train.npz", help='FID cache')

device = torch.device('cuda')

def custom_sort_key(file_name):
    # 使用正则表达式提取 float、A/B 以及数字部分
    match = re.match(r'(\d+\.\d+)_(A|B)_(\d+)\.jpg', file_name)

    if match:
        float_part = float(match.group(1))
        a_b_part = match.group(2)
        num_part = int(match.group(3))
        return (float_part, a_b_part, num_part)
    else:
        return file_name


def m_g():
    image_paths = glob.glob(os.path.join(FLAGS.similar_dir, '**/*.jpg'), recursive=True)
    file_name = [os.path.basename(path) for path in image_paths]
    sorted_files = sorted(file_name, key=custom_sort_key)
    # sorted_files = sorted(image_paths)
    # 定义转换器，将图像转为张量
    transform = transforms.Compose([
        transforms.Resize((FLAGS.img_size, FLAGS.img_size)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转为张量
    ])
    # 创建一个空列表用于存储张量化后的图像
    tensor_images = []
    k = 0
    num_of_img = 5
    # 遍历图像路径列表，将每张图像转换为张量并存储在 tensor_images 中
    for path in sorted_files:
        path = FLAGS.similar_dir + "/" + path
        k = k + 1
        if k > 256 * num_of_img:
            break
        img = Image.open(path)  # 打开图像文件
        img = transform(img)  # 将图像转为张量
        tensor_images.append(img)  # 存储张量化后的图像
    # exit(0)
    # 将列表转换为张量
    batch_images_tensor = torch.stack(tensor_images)
    for i in range(num_of_img):
        # 将一批图像张量（比如一个 minibatch）组合成一个网格状的图像
        # images_tensor 是一个 shape 为 (batch_size, channels, height, width) 的张量
        grid_img = make_grid(batch_images_tensor[i * 256:256 * (i + 1)], nrow=16, padding=2)
        to_pil = transforms.ToPILImage()
        # 将网格化的图像张量转换成 PIL 图像
        grid_pil = to_pil(grid_img)
        # 然后，使用 PIL 库的 save 方法保存图像到本地文件系统
        grid_pil.save(os.path.join(FLAGS.grid_dir, f'grid_image{i}.png'))


def main(argv):
    for flag_name in FLAGS:
        print(f'{flag_name}: {FLAGS[flag_name].value}')
    os.makedirs(FLAGS.similar_dir, exist_ok=True)  # 创建文件夹
    os.makedirs(FLAGS.grid_dir, exist_ok=True)  # 创建文件夹
    m_g()


if __name__ == '__main__':
    app.run(main)

