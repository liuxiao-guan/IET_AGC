import copy
import json
import os
import sys
import math
from random import sample
import warnings
from absl import app, flags
import re

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import numpy as np

from diffusion_fed import GaussianDiffusionTrainer, GaussianDiffusionSampler
from clients_fed_es import ClientsGroup
sys.path.append(r"/root/autodl-fs/Project/IET-AGC")
from model import UNet
from score.both import get_inception_and_fid_score
import random


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', True, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', [
                  'xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', [
                  'fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_enum('dataset', 'cifar10', ['tiny_imagenet', 'cifar10', 'celeba', 'afhq', 'cifar100','cifar10_rand'], help='dataset')
flags.DEFINE_float('mem_ratio', 2, help='')
flags.DEFINE_float('arr_ema_decay', 0.8, help='')
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_string('check_dir', None, help='check point dir')
# Fed
flags.DEFINE_integer('mid_T', 500, help='mid T split local global')
flags.DEFINE_bool('use_labels', False, help='use labels')
flags.DEFINE_integer('num_labels', None, help='num of classes')
flags.DEFINE_integer('local_epoch', 100, help='local epoch')
flags.DEFINE_integer('total_round', 10, help='total round')
flags.DEFINE_integer('client_num',10, help='client num')
flags.DEFINE_integer('save_round', 1, help='save round')
# Logging & Sampling
flags.DEFINE_string(
    'logdir', '/root/autodl-tmp/logs/Projects/IET-AGC/test_cifar10_cond_c10_v2', help='log directory')
flags.DEFINE_integer('sample_size', 100, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer(
    'save_step', 50000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer(
    'eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000,
                     help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_integer('seed', 12, help='random seed')

device = torch.device('cuda:0')

def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device),0,1000).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    seed_everything(seed=FLAGS.seed)
    # global_ckpt = torch.load(
    #     'logs/cifar10_mixT_500/global_ckpt_round1300.pt', map_location=torch.device('cpu'))
    # model setup
    net_model_global = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels)
    # log setup
    if not os.path.exists(FLAGS.logdir+'/sample'):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
        os.makedirs(os.path.join(FLAGS.logdir, 'sample', 'clients'))
    x_T = torch.randn(10, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    writer = SummaryWriter(FLAGS.logdir)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    clients_group = ClientsGroup(
        FLAGS.dataset, FLAGS.batch_size, FLAGS.client_num, device)
    # init local_parameters
    for i in range(FLAGS.client_num):
        clients_group.clients_set[i].init(net_model_global, FLAGS.lr, FLAGS.parallel,
                                          None if FLAGS.check_dir is None else torch.load(FLAGS.check_dir))
    client_idx = [x for x in range(FLAGS.client_num)]

    # here
    start = 0 if FLAGS.check_dir is None else int(re.findall(r'\d+', FLAGS.check_dir)[-1])
    index_array = np.zeros(150000, dtype=int)
    # start training
    from tqdm import tqdm
    import time
    for round in tqdm(range(FLAGS.total_round), desc="Training Progress"):
        sum_parameters = None
        sum_ema_parameters = None
        train_data_sum = 0
        for c in client_idx:
            train_data_sum += len(
                clients_group.clients_set[c].train_dataset.targets)
        # train
        for c in client_idx:
            
            global_parameters, global_ema_parameters = clients_group.clients_set[c].local_train(
                round, FLAGS.local_epoch, mid_T=FLAGS.mid_T, use_labels=FLAGS.use_labels,img_size = FLAGS.img_size,
                logdir = FLAGS.logdir,writer = writer,num_labels = FLAGS.num_labels, mem_ratio=FLAGS.mem_ratio,
                arr_ema_decay=FLAGS.arr_ema_decay, index_array=index_array)
            
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in global_parameters.items():
                    sum_parameters[key] = var.clone()
                    sum_parameters[key] = sum_parameters[key] / train_data_sum * len(
                        clients_group.clients_set[c].train_dataset.targets)
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + global_parameters[var] / train_data_sum * len(
                        clients_group.clients_set[c].train_dataset.targets)

            if sum_ema_parameters is None:
                sum_ema_parameters = {}
                for key, var in global_ema_parameters.items():
                    sum_ema_parameters[key] = var.clone()
                    sum_ema_parameters[key] = sum_ema_parameters[key] / train_data_sum * len(
                        clients_group.clients_set[c].train_dataset.targets)
            else:
                for var in sum_ema_parameters:
                    sum_ema_parameters[var] = sum_ema_parameters[var] + global_ema_parameters[var] / train_data_sum * len(
                        clients_group.clients_set[c].train_dataset.targets)
        # fedavg
        for c in client_idx:
            clients_group.clients_set[c].set_global_parameters(
                sum_parameters, sum_ema_parameters)
        # sample
        samples = []
        for c in client_idx:
            client = clients_group.clients_set[c]
            client.global_ema_model.eval()
            with torch.no_grad():
                if FLAGS.num_labels is None:
                    x_0 = client.global_ema_sampler(x_T, 0, 1000)
                else:
                    labels = []
                    for label in range(FLAGS.num_labels):
                        labels.append(torch.ones(1, dtype=torch.long, device=device) * label)
                    labels = torch.cat(labels, dim=0)
                    x_0 = client.global_ema_sampler(x_T, 0, 1000)
                samples.append(x_0)
            client.global_ema_model.train()
        samples = torch.cat(samples, dim=0)
        grid = (make_grid(samples, nrow=10) + 1) / 2
        path = os.path.join(
            FLAGS.logdir, 'sample', f'{round+1}.png')
        save_image(grid, path)
        writer.add_image('sample', grid, round+1)

        # save
        if FLAGS.save_round > 0 and (round+1) % FLAGS.save_round == 0:
            # save global_model
            global_ckpt = {
                'global_model': sum_parameters,
                'global_ema_model': sum_ema_parameters,
                'index_array': index_array,
            }
            global_model_path = FLAGS.logdir
            torch.save(global_ckpt, os.path.join(
                global_model_path, 'global_ckpt_round{}.pt'.format(round+1)))
    writer.close()

def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(model)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'global_ckpt_round10.pt'))
    model.load_state_dict(ckpt['global_model'])
    ema_model.load_state_dict(ckpt['global_ema_model'])
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # load model and evaluate
    #ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

   
    (IS, IS_std), FID, samples = evaluate(ema_sampler, ema_model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)

def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
