import copy
import json
import os
import math
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10,CIFAR100
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score
from torchvision import transforms, datasets
from utils.dataset_utils import load_data
from utils.imagenet64 import ImageNet64
from data_utils import Crop, celeba,__init__
from torchvision.transforms.functional import crop
from torchvision.datasets import ImageFolder

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', True, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('num_labels', None, help='num of classes')
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
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
flags.DEFINE_integer('total_steps', 400000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size',128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_enum('dataset','afhq',['tiny_imagenet','cifar10','imagenet2012','imagenet64','cifar100','celeba','afhq'],help='dataset')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/celeba/central_3000', help='log directory')
flags.DEFINE_integer('sample_size', 100, "sampling size of images")
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 10000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for data in iter(dataloader):
            yield data


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


# def lr_sched(step):
#     if step<=FLAGS.warmup:
#         return step / FLAGS.warmup
#     else:
#         return 0.5 * (math.cos((step - FLAGS.warmup) /(FLAGS.total_steps - FLAGS.warmup) * math.pi) + 1)
        

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
                labels = []
                import random
                    # 生成十个随机数字
                random_labels =  [random.randint(1, 100) for _ in range(batch_size)]
                for label in random_labels:
                    labels.append(torch.ones(1, dtype=torch.long, device=device) * label)
                labels = torch.cat(labels, dim=0)
                batch_images = sampler(x_T.to(device), labels.to(device)).cpu()
            #batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    if FLAGS.dataset == 'cifar10':
        # dataset
        dataset = CIFAR10(
            root='/root/autodl-pub/datasets/dataset/', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True)
    elif FLAGS.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        dataset = CIFAR100(root='/data1/gxl/dataset/', train=True,download=True,
                               transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset , batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    elif FLAGS.dataset == 'celeba':
        subset_indics = [i for i in range(0,300)]
        dataset = celeba.CelebA(
                root='/data3_u2/gxl/dataset', split='train', download=False,indics = subset_indics ,
                transform=transforms.Compose([
                    transforms.Lambda(lambda x: crop(x, top=57, left=25, height=128, width=128)),
                    transforms.Resize(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
              )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True)
    elif FLAGS.dataset == 'afhq':
        transform = transforms.Compose([
            transforms.Resize((FLAGS.img_size, FLAGS.img_size)),  # 调整图像大小
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
        ])
        train_dataset = ImageFolder(root='/data3_u2/gxl/dataset/afhq/train/dog', transform=transform)
        if FLAGS.num_train_img:
            num_train_img = FLAGS.num_train_img
        else:
            num_train_img = len(train_dataset)
        subset_sampler = torch.utils.data.SubsetRandomSampler(range(num_train_img))
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                sampler=subset_sampler)
    datalooper = infiniteloop(dataloader)
     
    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels)
    # # load model and evaluate
    # ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt20k.pt'))
    # net_model.load_state_dict(ckpt['ema_model'])
    ema_model = copy.deepcopy(net_model)
    
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    if not os.path.exists(FLAGS.logdir+'/sample'):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0, labels = next(datalooper)
            x_0 = x_0.to(device)
            labels = labels.to(device)
            if FLAGS.num_labels is None:
                loss = trainer(x_0).mean()
            else:
                loss = trainer(x_0, labels).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss, lr='%.6f' % sched.get_last_lr()[-1])

            # sample
            if FLAGS.sample_step > 0 and (step+1) % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    if FLAGS.num_labels is None:
                        x_0 = ema_sampler(x_T)
                    else:
                        labels = []
                        for label in range(FLAGS.num_labels):
                            labels.append(torch.ones(100//FLAGS.num_labels, dtype=torch.long, device=device) * label)
                        labels = torch.cat(labels, dim=0)
                        x_0 = ema_sampler(x_T, labels)
                    grid = (make_grid(x_0, nrow=10) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and (step+1) % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt{}k.pt'.format(step//1000)))

            # evaluate
            # if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
            #     net_IS, net_FID, _ = evaluate(net_sampler, net_model)
            #     ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
            #     metrics = {
            #         'IS': net_IS[0],
            #         'IS_std': net_IS[1],
            #         'FID': net_FID,
            #         'IS_EMA': ema_IS[0],
            #         'IS_std_EMA': ema_IS[1],
            #         'FID_EMA': ema_FID
            #     }
            #     pbar.write(
            #         "%d/%d " % (step, FLAGS.total_steps) +
            #         ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
            #     for name, value in metrics.items():
            #         writer.add_scalar(name, value, step)
            #     writer.flush()
            #     with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
            #         metrics['step'] = step
            #         f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,num_labels=FLAGS.num_labels)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt180k.pt'))
    model.load_state_dict(ckpt['net_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    for flag_name in FLAGS:
        print(f'{flag_name}: {FLAGS[flag_name].value}')
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
