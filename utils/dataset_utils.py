import os
from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names))) if x != 'ILSVRC2012'}
        classes = [sorted_classes[x] for x in class_names]
    
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=1,
        num_shards=0,
        index=0
        # shard=MPI.COMM_WORLD.Get_rank(),
        # num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    class_labels = [
    'n01558993', 'n01855672', 'n02093256', 'n02100877', 'n02114855', 'n02325366',
    'n02906734', 'n03272010', 'n03623198', 'n03814639', 'n03956157', 'n04235860',
    'n04443257', 'n04591157', 'n07753592', 'n01601694', 'n01871265', 'n02093754',
    'n02104365', 'n02120079', 'n02364673', 'n02909870', 'n03291819', 'n03649909',
    'n03837869', 'n03983396', 'n04311004', 'n04458633', 'n04592741', 'n11879895',
    'n01669191', 'n02018207', 'n02094114', 'n02105855', 'n02120505', 'n02484975',
    'n03085013', 'n03337140', 'n03710721', 'n03838899', 'n04004767', 'n04325704',
    'n04483307', 'n04606251', 'n01751748', 'n02037110', 'n02096177', 'n02106030',
    'n02125311', 'n02489166', 'n03124170', 'n03450230', 'n03717622', 'n03854065',
    'n04026417', 'n04336792', 'n04509417', 'n07583066', 'n01755581', 'n02058221',
    'n02097130', 'n02106166', 'n02128385', 'n02708093', 'n03127747', 'n03483316',
    'n03733281', 'n03929855', 'n04065272', 'n04346328', 'n04515003', 'n07613480',
    'n01756291', 'n02087046', 'n02097298', 'n02107142', 'n02133161', 'n02747177',
    'n03160309', 'n03498962', 'n03759954', 'n03930313', 'n04200800', 'n04380533',
    'n04525305', 'n07693725', 'n01770393', 'n02088632', 'n02099267', 'n02110341',
    'n02277742', 'n02835271', 'n03255030', 'n03530642', 'n03775071', 'n03954731',
    'n04209239', 'n04428191', 'n04554684', 'n07711569'
    ]

    select_label = [72, 62, 52, 49, 11, 10, 13, 80, 39, 25, 90, 5, 29, 21, 77, 97, 65, 37, 28, 
              14, 70, 6, 99, 88, 79, 32, 64, 83, 33, 19, 78, 31, 93, 20, 86, 56, 74, 55, 
              61, 42, 35, 1, 16, 22, 60, 67, 87, 91, 76, 71]
    selected_classes = [class_labels[index] for index in select_label]
    for entry in sorted(bf.listdir(data_dir)):
        if entry not in selected_classes and os.path.basename(data_dir) not in selected_classes:
            continue
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1,index=0):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.local_classes = classes
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
                out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict['y']