import logging

from PIL import Image
import torch
import numpy as np

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)




def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def crop_arr(pil_image, max_image_size):
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1

    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1

    arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]    
    return Image.fromarray(arr)



def vae_encode(vae, x, weight_dtype):
    if x is not None:
        if vae.config.shift_factor is not None:
            x = vae.encode(x).latent_dist.sample()
            x = (x - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
        x = x.to(weight_dtype)
    return x

def vae_encode_list(vae, x, weight_dtype):
    latents = []
    for img in x:
        img = vae_encode(vae, img, weight_dtype)
        latents.append(img)
    return latents

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def concatenate_images(images, nrows, ncols,reversed = False, padding=0, background_color=(255, 255, 255)):
    """
    拼接图像列表为一个大图像。

    参数:
    images (list of PIL.Image): 要拼接的图像列表。
    nrows (int): 行数。
    ncols (int): 列数。
    padding (int): 图像之间的填充宽度（默认为0）。
    background_color (tuple): 填充区域的颜色（默认为白色）。

    返回:
    PIL.Image: 拼接后的图像。
    """
    total_images = nrows * ncols
    if len(images) < total_images:
        # 如果图像数量不足，创建空白图像填补空缺
        image_size = images[0].size if images else (100, 100)  # 默认大小为 100x100
        blank_images = [Image.new('RGB',image_size, background_color) for _ in range(total_images - len(images))]
        images.extend(blank_images)

    # 获取所有图像的宽度和高度
    widths, heights = zip(*(i.size for i in images))

    # 计算拼接后图像的总宽度和高度
    total_width = max(widths) * ncols + padding * (ncols - 1)
    total_height = max(heights) * nrows + padding * (nrows - 1)

    # 创建一个新的空白图像
    new_img = Image.new('RGB', (total_width, total_height), color=background_color)
    if not reversed:    
        # 将每个图像粘贴到新的图像中
        for i, img in enumerate(images):
            row = i // ncols
            col = i % ncols
            x_offset = col * (max(widths) + padding)
            y_offset = row * (max(heights) + padding)
            new_img.paste(img, (x_offset, y_offset))
    else:
        for i, img in enumerate(images):
            row = i % nrows
            col = i // nrows
            x_offset = col * (max(widths) + padding)
            y_offset = row * (max(heights) + padding)
            new_img.paste(img, (x_offset, y_offset))
    return new_img



masks = {}
frac_factors = {}
intermediate_vars = {}

def find_element_indices(lst, target):
    """
    查找列表中所有等于目标元素的索引。
    
    参数:
    - lst: 输入列表
    - target: 要查找的目标元素
    
    返回:
    - 目标元素的所有出现位置的索引列表
    """
    indices = [index for index, element in enumerate(lst) if element == target]
    return indices

def print_cuda_memory(device=0):
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return

    device = torch.device(f"cuda:{device}")
    print(f"统计 GPU {device} 的显存使用情况:")

    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    print(f"当前已分配 (allocated): {allocated:.2f} MB")
    print(f"当前已保留 (reserved): {reserved:.2f} MB")
    print(f"最大分配记录 (max allocated): {max_allocated:.2f} MB")
    print(f"最大保留记录 (max reserved): {max_reserved:.2f} MB")
