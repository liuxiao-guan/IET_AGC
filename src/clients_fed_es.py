import copy
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm, trange
from diffusion_fed import GaussianDiffusionTrainer, GaussianDiffusionSampler
from torchvision.utils import make_grid, save_image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import Dataset

sys.path.append(r"/root/autodl-fs/Project/IET-AGC")
from model import UNet
from data_utils import Crop, celeba,__init__


class IndexedCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super(IndexedCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(IndexedCIFAR10, self).__getitem__(index)
        return img, target, index


class IndexedCIFAR100(CIFAR100):
    def __init__(self, *args, **kwargs):
        super(IndexedCIFAR100, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(IndexedCIFAR100, self).__getitem__(index)
        return img, target, index


class CustomImageDataset(Dataset):
    def __init__(self, root, index_list, transform=None):
        self.image_folder = ImageFolder(root, transform)
        self.index_list = index_list
        self.targets = [self.image_folder.targets[i] for i in index_list]

    def __getitem__(self, index):
        original_index = self.index_list[index]
        img, target = self.image_folder[original_index]
        return img, target, index

    def __len__(self):
        return len(self.index_list)


class IndexedCeleba(celeba.CelebA):
    def __init__(self, *args, **kwargs):
        super(IndexedCeleba, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(IndexedCeleba, self).__getitem__(index)
        return img, target, index


class Client(object):
    def __init__(self, client_id, train_dataset, train_loader, device):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.device = device

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None

        self._step_cound = 0
        self.avg_loss = torch.zeros(1000)

    def warmup_lr(self, step):
        warmup_epoch = 15
        warmup_iters = len(self.train_loader) * warmup_epoch
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def init(self, model_global, lr, parallel, global_ckpt=None):
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)
        self.global_trainer = GaussianDiffusionTrainer(
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        self.global_ema_sampler = GaussianDiffusionSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(self.device)

        if parallel:
            self.global_trainer = torch.nn.DataParallel(self.global_trainer)
            self.global_ema_sampler = torch.nn.DataParallel(self.global_ema_sampler)
        self.avg_loss = self.avg_loss.to(self.device)
    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)

    def local_train(self, round, local_epoch, mid_T, use_labels=True, img_size=32, logdir=None, writer=None,
                    num_labels=None, mem_ratio=2., arr_ema_decay=0.8, index_array=None):
        self.global_trainer.train()
        global_loss = 0
        for epoch in range(local_epoch):
            with tqdm(self.train_loader, dynamic_ncols=True,
                      desc=f'round:{round + 1} client:{self.client_id} epoch:{epoch + 1}') as pbar:
                for x, label, indexes in pbar:
                    x, label = x.to(self.device), label.to(self.device)
                    t = torch.randint(0, 1000, size=(x.shape[0],), device=x.device)
                    if use_labels:
                        ori_loss = torch.mean(self.global_trainer(x, t, 0, 1000, label), dim=(1, 2, 3))
                    else:
                        ori_loss = torch.mean(self.global_trainer(x, t, 0, 1000), dim=(1, 2, 3))
                    ori_loss = ori_loss.to(x.device)
                    mean_loss = self.avg_loss[t]
                    mean_loss = mean_loss.to(x.device)
                    mask = ori_loss * mem_ratio >= mean_loss.float().detach().to(x.device)
                    tot = max(mask.sum().item(), 1)
                    global_loss = ((ori_loss * mask) * (x.shape[0] / tot)).mean()
                    #index_array[indexes[mask == 0]] += 1
                    for index, timestep in enumerate(t):
                        self.avg_loss[timestep] = arr_ema_decay * self.avg_loss[timestep] + (1. - arr_ema_decay) * ori_loss[
                            index].item()
                    # global update
                    self.global_optim.zero_grad()
                    global_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.global_optim.step()
                    self.global_sched.step()
                    self.ema(self.global_model, self.global_ema_model, 0.9999)

                    # log
                    pbar.set_postfix(global_loss='%.3f' % global_loss, lr='%.6f' % self.global_sched.get_last_lr()[-1])
                    self._step_cound += 1

            # sample
            if epoch + 1 == local_epoch or (epoch + 1) * 2 == local_epoch:
                x_T = torch.randn(10, 3, img_size, img_size)
                x_T = x_T.to(self.device)
                self.global_ema_model.eval()
                with torch.no_grad():
                    if num_labels is None:
                        x_0 = self.global_ema_sampler(x_T, 0, 1000)
                    else:
                        labels = []
                        for label in range(num_labels):
                            labels.append(torch.ones(10, dtype=torch.long, device=self.device) * label)
                        labels = torch.cat(labels, dim=0)
                        x_0 = self.global_ema_sampler(x_T, 0, 1000)
                    grid = (make_grid(x_0, nrow=10) + 1) / 2
                    os.makedirs(os.path.join(logdir, 'sample', 'clients'),exist_ok=True)
                    path = os.path.join(
                        logdir, 'sample', 'clients', f'{round + 1}_{self.client_id}_{epoch}.png')
                    save_image(grid, path)
                    writer.add_image('sample', grid, round + 1)
                self.global_ema_model.train()

        return self.global_model.state_dict(), self.global_ema_model.state_dict()


class ClientsGroup(object):

    def __init__(self, dataset_name, batch_size, clients_num, device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.clients_num = clients_num
        self.device = device
        self.clients_set = []
        self.test_loader = None
        self.data_allocation()

    def data_allocation(self):
        if self.dataset_name == 'cifar10':
            # cifar10
            train_dataset = IndexedCIFAR10(
                root='/root/autodl-pub/datasets/dataset/',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
            clients_train_data_idxs = [[] for i in range(len(train_dataset.classes))]
            for idx, target in enumerate(train_dataset.targets):
                clients_train_data_idxs[target].append(idx)
            clients_train_data_idxs = np.array(
                list(map(np.array, clients_train_data_idxs)))
            for i in range(self.clients_num):
                train_dataset_client = IndexedCIFAR10(
                    root='/root/autodl-pub/datasets/dataset/',
                    train=True,
                    download=False,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
                # 2 class per client
                # client_data_idxs = np.concatenate(
                #     clients_train_data_idxs[2*i:2*i+2])
                # iid per client
                num_per_class = 5000 // self.clients_num
                client_data_idxs = np.concatenate(
                    clients_train_data_idxs[:, num_per_class * i: num_per_class * (i + 1)])
                train_dataset_client.data = train_dataset_client.data[client_data_idxs]
                train_dataset_client.targets = np.array(train_dataset_client.targets)[
                    client_data_idxs].tolist()
                train_loader_client = DataLoader(
                    train_dataset_client,
                    batch_size=self.batch_size,
                    sampler=torch.utils.data.RandomSampler(train_dataset_client),
                    drop_last=True,
                    num_workers=4)
                client = Client(i, train_dataset_client,
                                train_loader_client, self.device)
                self.clients_set.append(client)

        elif self.dataset_name == 'cifar100':
            train_dataset = IndexedCIFAR100(
                root='/data3_u2/lx/dataset',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
            clients_train_data_idxs = [[] for i in range(len(train_dataset.classes))]
            for idx, target in enumerate(train_dataset.targets):
                clients_train_data_idxs[target].append(idx)
            clients_train_data_idxs = np.array(
                list(map(np.array, clients_train_data_idxs)))

            for i in range(self.clients_num):
                train_dataset_client = IndexedCIFAR100(
                    root='/data3_u2/lx/dataset',
                    train=True,
                    download=False,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
                # 2 class per client
                # client_data_idxs = np.concatenate(
                #     clients_train_data_idxs[2*i:2*i+2])
                # iid per client
                client_data_idxs = np.concatenate(
                clients_train_data_idxs[:, 50 * i:50 * (i + 1)])
                train_dataset_client.data = train_dataset_client.data[client_data_idxs]
                train_dataset_client.targets = np.array(train_dataset_client.targets)[
                    client_data_idxs].tolist()
                train_loader_client = DataLoader(
                    train_dataset_client,
                    batch_size=self.batch_size,
                    sampler=torch.utils.data.RandomSampler(train_dataset_client),
                    drop_last=True,
                    num_workers=4)
                client = Client(i, train_dataset_client,
                                train_loader_client, self.device)
                self.clients_set.append(client)

        elif self.dataset_name == 'afhq':
            transform = transforms.Compose([
                transforms.Resize((64, 64)),  # 调整图像大小
                transforms.ToTensor(),  # 将图像转换为张量
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
            ])
            from torch.utils.data import random_split
            train_dataset_len = 4739
            subset_size = 4739 // self.clients_num
            # 将数据集随机分割成十个子集的索引
            indices = list(range(train_dataset_len))
            subsets_indices = [indices[i * subset_size: (i + 1) * subset_size] for i in range(self.clients_num)]
            for i in range(self.clients_num):
                train_dataset_client = CustomImageDataset(root='/data3_u2/lx/dataset/afhq', index_list=subsets_indices[i],
                                                            transform=transform)
                train_loader_client = DataLoader(
                    train_dataset_client,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4)
                client = Client(i, train_dataset_client,
                                train_loader_client, self.device)
                self.clients_set.append(client)
        elif self.dataset_name == 'cifar10_rand':
            # cifar10
            train_dataset = IndexedCIFAR10(
                root='/root/autodl-pub/datasets/dataset/',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
            # 将数据集打乱
            indices = np.arange(50000)
            np.random.shuffle(indices)
            sublists = np.array_split(indices, self.clients_num)
            client_indices_list = [sublist.tolist() for sublist in sublists]
            for i in range(self.clients_num):
                train_dataset_client = IndexedCIFAR10(
                    root='/root/autodl-pub/datasets/dataset/',
                    train=True,
                    download=False,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
                # 2 class per client
                # client_data_idxs = np.concatenate(
                #     clients_train_data_idxs[2*i:2*i+2])
                # iid per client
                
                train_dataset_client.data = train_dataset_client.data[sublists[i]]
                train_dataset_client.targets = np.array(train_dataset_client.targets)[
                    sublists[i]].tolist()
                train_loader_client = DataLoader(
                    train_dataset_client,
                    batch_size=self.batch_size,
                    sampler=torch.utils.data.RandomSampler(train_dataset_client),
                    drop_last=True,
                    num_workers=4)
                client = Client(i, train_dataset_client,
                                train_loader_client, self.device)
                self.clients_set.append(client)
           