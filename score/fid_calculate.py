import os

import numpy as np
import torch
import types
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import crop
# import cv2
from PIL import Image
from torchvision.datasets import CIFAR10, CelebA, ImageFolder
import torchvision
import sys
from inception import InceptionV3
from fid import calculate_frechet_distance, torch_cov

sys.path.append(r"/home/gxl/Project/IET-AGC")
from data_utils import celeba


# from utils.imagenet64 import ImageNet64



device = torch.device('cuda')


def get_inception_and_fid_score(images, num_images=None,
                                batch_size=50,
                                verbose=False,
                                parallel=False):
    """when `images` is a python generator, `num_images` should be given"""

    if num_images is None and isinstance(images, types.GeneratorType):
        raise ValueError(
            "when `images` is a python generator, "
            "`num_images` should be given")

    if num_images is None:
        num_images = len(images)

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    fid_acts = np.empty((num_images, 2048))
    is_probs = np.empty((num_images, 1008))

    iterator = iter(tqdm(
        images, total=num_images,
        dynamic_ncols=True, leave=False, disable=not verbose,
        desc="get_inception_and_fid_score"))

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            fid_acts[start: end] = pred[0].view(-1, 2048).cpu().numpy()
            is_probs[start: end] = pred[1].cpu().numpy()
        start = end

    m1 = np.mean(fid_acts, axis=0)
    s1 = np.cov(fid_acts, rowvar=False)
    # print(m1)
    # print(s1)

    np.savez('./stats/celeba64_3000.train.npz', mu=m1[:], sigma=s1[:])

    # del fid_acts, is_probs, model
    return


def get_dataset_npy():
    subset_indics = [i for i in range(0,3000)]
    dataset = celeba.CelebA(
            root='/data3_u2/gxl/dataset', split='train', download=False,indics = subset_indics ,
            transform=transforms.Compose([
                transforms.Lambda(lambda x: crop(x, top=57, left=25, height=128, width=128)),
                transforms.Resize(64),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            )
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True,
            num_workers=4, drop_last=False)
    # transform_train = transforms.Compose([
    #     transforms.Resize(64),
    #     transforms.ToTensor()
    # ])
    # train_dataset = ImageFolder(root='/data3_u2/lx/dataset/afhq', transform=transform_train)
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    # dataset = CelebA(
    #     root='/home/lx/IET-AGC/data', split='train', download=True,
    #     transform=transforms.Compose([
    #         transforms.Lambda(lambda x: crop(x, top=57, left=25, height=128, width=128)),
    #         transforms.Resize(64),  # 请根据需要调整大小
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    # )
    # print(len(dataset))
    #
    # 数据加载器
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
    # 将图像转换为 numpy 数组并存储为 .npy 文件
    image_array = []
    for batch in dataloader:
        images = batch[0]  # 这里假设 batch 的第一个元素是图像
        image_array.append(images.numpy())
    # 将列表转换为 numpy 数组
    image_array = np.concatenate(image_array, axis=0)
    print(image_array.shape)
    print(np.max(image_array))
    print(np.min(image_array))
    # 保存为 .npy 文件
    np.save('./stats/celeba64_3000.npy', image_array)


if __name__ == '__main__':
    get_dataset_npy()

    images_array = np.load('./stats/celeba64_3000.npy')
    get_inception_and_fid_score(images_array)

    # f2 = np.load("/home/lx/jupyter-notebook/IET-AGC/stats/cifar10.train.npz")
    # m2, s2 = f2['mu'][:], f2['sigma'][:]
    # print(m2)
    # print(s2)
    # print("------")
    #
    # f1 = np.load("/home/lx/jupyter-notebook/IET-AGC/stats/mycifar10.train.npz")
    # m1, s1 = f1['mu'][:], f1['sigma'][:]
    # print(m1)
    # print(s1)

    #
    # root_folder = '/home/lx/jupyter-notebook/IET-AGC/data/tiny-imagenet-200/train'
    #
    # # 定义图像变换
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # 将图像转换为 PyTorch 的张量
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    # ])
    #
    # # 读取文件夹下所有图片并进行变换
    # images = []
    #
    # for folder_name in os.listdir(root_folder):
    #     folder_path = os.path.join(root_folder, folder_name, "images")
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith(('.jpg', '.JPEG', '.png')):  # 可以根据需要添加其他图像格式的扩展名
    #             image_path = os.path.join(folder_path, filename)
    #
    #             # 使用 OpenCV 读取图像
    #             image_cv2 = cv2.imread(image_path)
    #
    #             # 转换为 RGB 模式（OpenCV 默认为 BGR）
    #             image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    #
    #             # 转换为 PIL Image
    #             image_pil = Image.fromarray(image_rgb)
    #
    #             # 应用变换
    #             transformed_image = transform(image_pil)
    #
    #             # 将变换后的图像添加到列表中
    #             images.append((transformed_image.numpy()+1) / 2)
    #             print(np.min((transformed_image.numpy()+1) / 2))
    #             exit(0)
    #
    #
    # # 将列表转换为 NumPy 数组
    # images_array = np.array(images)
    # print(images_array.shape)
    # save_path = '/home/lx/jupyter-notebook/IET-AGC/stats/tmp.npy'
    #
    # # 保存 images_array 到 .npy 文件
    # np.save(save_path, images_array)
    #
    #
    #
