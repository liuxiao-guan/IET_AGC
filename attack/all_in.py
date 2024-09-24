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


def maintain_array(values, current_array):
    combined_array = current_array + values
    sorted_array = sorted(combined_array)[:50]
    return sorted_array


class CustomImageDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_paths = [os.path.join(data_folder, img) for img in os.listdir(data_folder) if
                            os.path.isfile(os.path.join(data_folder, img))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 载入图像并确保是RGB格式

        # 提取图像文件名
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]

        if self.transform:
            image = self.transform(image)

        return image, img_name


class ImageFolderWithName(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # 调用父类方法获取图像和标签
        path, _ = self.samples[index]  # 获取图像路径

        # 提取图像文件名
        img_name = os.path.basename(path)
        img_name = os.path.splitext(img_name)[0]

        return img, img_name, target, path


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            if FLAGS.num_labels is None:
                batch_images = sampler(x_T.to(device)).cpu()
            else:
                labels = torch.randint(0, 100, size=(batch_size,))
                batch_images = sampler(x_T.to(device), labels.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    # return (IS, IS_std), FID, images
    print("(IS, IS_std), FID: ", (IS, IS_std), FID)
    with open(FLAGS.grid_dir + "/result.txt", 'a') as file:
        # 写入内容
        file.write(f"(IS, IS_std), FID: ({IS}, {IS_std}), {FID}")
    np.save(FLAGS.generate_dir+"/generate.npy", images)
    return images


def generate():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    if FLAGS.Fed:
        # print(os.getcwd())
        ckpt = torch.load(FLAGS.unet_dir)
        model.load_state_dict(ckpt['global_model'])
    else:
        # print(os.path.join(logdir, 'ckpt439k_tinyimagenet.pt'))
        ckpt = torch.load(FLAGS.unet_dir)
        if FLAGS.use_ema:
            model.load_state_dict(ckpt['ema_model'])
            print("ema")
        else:
            model.load_state_dict(ckpt['net_model'])
            print("net")

    os.makedirs(FLAGS.generate_dir, exist_ok=True)
    samples = evaluate(sampler, model)

    for i in range(len(samples)):
        save_image(torch.tensor(samples[i]), os.path.join(FLAGS.generate_dir, '%d.png' % i))


def compute_loss(data_loader_A, data_loader_B):
    size_b = data_loader_B.shape[0]
    batch_size_B = FLAGS.batch_B
    name_B = 0

    for i in tqdm(range(int(size_b / batch_size_B))):
        tensor_B = data_loader_B[i * batch_size_B: (i + 1) * batch_size_B]
        tensor_B = torch.from_numpy(tensor_B)
        tensor_B = tensor_B.reshape(batch_size_B, -1)
        tensor_B = tensor_B.to(device)
        # dis = []
        min_loss = [10000 for _ in range(batch_size_B)]

        # here
        min_loss_top50 = [[] for _ in range(batch_size_B)]

        image_A = [[] for _ in range(batch_size_B)]
        for tensor_A, _ in data_loader_A:
            tmp_b = tensor_B.unsqueeze(1).expand(-1, tensor_A.size(0), -1)
            tensor_A_flat = tensor_A.to(device)
            tensor_A_flat = tensor_A_flat.view(tensor_A_flat.size(0), -1)
            l2_loss = F.mse_loss(tmp_b, tensor_A_flat, reduction='none')
            l2_loss = torch.sum(l2_loss, dim=-1)
            tmp_min_loss, tmp_min_index = torch.min(l2_loss, dim=-1)

            for j in range(batch_size_B):
                tmp = tmp_min_loss[j].item()
                if tmp < min_loss[j]:
                    min_loss[j] = tmp
                    image_A[j] = tensor_A[tmp_min_index[j].item()]
                min_loss_top50[j] = maintain_array(min_loss_top50[j], l2_loss[j].tolist())

        # here
        least = torch.tensor(min_loss_top50)
        mean_values = torch.mean(least, dim=1)
        min_loss = torch.div(torch.tensor(min_loss), mean_values)
        name_B = name_B + 1
        attack_distance = min_loss
        image_A = torch.stack(image_A).view(batch_size_B, 3, FLAGS.img_size, FLAGS.img_size).permute(0, 2, 3, 1).cpu().numpy()
        tensor_B = tensor_B.view(batch_size_B, 3, FLAGS.img_size, FLAGS.img_size).permute(0, 2, 3, 1).cpu().numpy()
        save_image_pair(image_A, tensor_B, name_B, attack_distance)


def save_image_pair(imageA, imageB, name_B, attack_distance):
    print(100*"*")
    print(FLAGS.similar_dir)
    for i in range(FLAGS.batch_B):
        plt.imsave(os.path.join(FLAGS.similar_dir, f'{attack_distance[i]}_A_{name_B}.jpg'),
                   imageA[i])
        plt.imsave(os.path.join(FLAGS.similar_dir, f'{attack_distance[i]}_B_{name_B}.jpg'),
                   imageB[i])


def l2_compute():
    # here:dataset
    if FLAGS.dataset == 'cifar10':
        # 加载所有图像A
        dataset = CIFAR10(
            root='/root/autodl-pub/datasets/dataset/', train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize((FLAGS.img_size, FLAGS.img_size)),
                transforms.ToTensor(),
            ]))
        data_loader_A = DataLoader(dataset, batch_size=FLAGS.batch_A, shuffle=False)
    elif FLAGS.dataset == 'afhq':
        train_dataset = ImageFolder(root='/data3_u2/lx/dataset/afhq',
                                    transform=transforms.Compose([
                                        transforms.Resize((FLAGS.img_size, FLAGS.img_size)),  # 调整图像大小
                                        transforms.ToTensor(),  # 将图像转换为张量
                                    ]))
        data_loader_A = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_A, shuffle=False,
                                                    num_workers=FLAGS.num_workers)
    elif FLAGS.dataset == 'cifar100':
        dataset = CIFAR100(root='/data3_u2/gxl/dataset', train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize((FLAGS.img_size, FLAGS.img_size)),
                               transforms.ToTensor(),
                           ]))
        data_loader_A = DataLoader(dataset, batch_size=FLAGS.batch_A, shuffle=False)

    data_loader_B = np.load(FLAGS.generate_dir+"/generate.npy")
    data_loader_B = np.minimum(data_loader_B, 1)
    data_loader_B = np.maximum(data_loader_B, 0)
    compute_loss(data_loader_A, data_loader_B)


def count():
    decimal_pattern = re.compile(r'\d+\.\d+')
    v_15 = 0
    v_20 = 0
    v_30 = 0
    v_40 = 0
    v_50 = 0
    v_60 = 0
    v_70 = 0
    v_80 = 0
    v_90 = 0
    for filename in os.listdir(FLAGS.similar_dir):
        if decimal_pattern.search(filename):
            decimal_value = float(decimal_pattern.search(filename).group()) * 100
            if decimal_value < 15:
                v_15 += 1
            if decimal_value < 20:
                v_20 += 1
            if decimal_value < 30:
                v_30 += 1
            if decimal_value < 40:
                v_40 += 1
            if decimal_value < 50:
                v_50 += 1
            if decimal_value < 60:
                v_60 += 1
            if decimal_value < 70:
                v_70 += 1
            if decimal_value < 80:
                v_80 += 1
            if decimal_value < 90:
                v_90 += 1
    print(f"v_15:{v_15/2}")
    print(f"v_20:{v_20/2}")
    print(f"v_30:{v_30/2}")
    print(f"v_40:{v_40/2}")
    print(f"v_50:{v_50/2}")
    print(f"v_60:{v_60/2}")
    print(f"v_70:{v_70/2}")
    print(f"v_80:{v_80/2}")
    print(f"v_90:{v_90/2}")
    with open(FLAGS.grid_dir+"/result.txt", 'a') as file:
        # 写入内容
        file.write(f"v_15:{v_15/2}\n")
        file.write(f"v_20:{v_20/2}\n")
        file.write(f"v_30:{v_30/2}\n")
        file.write(f"v_40:{v_40/2}\n")
        file.write(f"v_50:{v_50/2}\n")
        file.write(f"v_60:{v_60/2}\n")
        file.write(f"v_70:{v_70/2}\n")
        file.write(f"v_80:{v_80/2}\n")
        file.write(f"v_90:{v_90/2}\n")


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
    generate()
    l2_compute()
    count()
    m_g()


if __name__ == '__main__':
    app.run(main)
