import numpy as np
import random
import torch
from torchvision import transforms
from PIL import ImageFilter

# 设置随机数种子
def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
# 基于KNN的高斯模糊操作
class KNNGaussianBlur(torch.nn.Module):
    # radius 表示高斯模糊的半径
    def __init__(self, radius : int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        return final_map
