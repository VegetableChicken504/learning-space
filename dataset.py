import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np

def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',   
    ]

def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

RGB_SIZE = 224
# 该类用于加载和处理异常检测任务中的图像数据集
class BaseAnomalyDetectionDataset(Dataset):

    def __init__(self, split, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        """
        split:用用于指定数据集中的哪个数据拆分（例如训练集、验证集、测试集）
        """
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        """
        self.IMAGENET_MEAN（均值） 和 self.IMAGENET_STD（标准差） 是用于归一化图像数据的常数
        """
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),  # 将输入图像调整为指定的大小（RGB_SIZE,RGB_SIZE），并使用BICUBIC插值来实现平滑的大小调整
             transforms.ToTensor(),  # 转换为张量
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])  # 进行均值和标准差归一化
# 该类将预训练的张量数据加载到PyTorch数据集中
class PreTrainTensorDataset(Dataset):
    # root_path:指包含预训练张量数据的根路径
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.tensor_paths)
    # 获取数据集中特定索引位置idx处的样本
    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.root_path, tensor_path))
        # 为每个加载的张量分配标签
        label = 0
        # 返回加载的张量和相应的标签作为数据集的一个样本
        return tensor, label

class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        # 首先调用基类的构造函数，并设置了一些成员变量
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    # 加载数据集中的图像路径和标签
    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        # glob.glob（）函数找到rgb图像和tiff文件的路径
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        # sort()函数排序
        rgb_paths.sort()
        tiff_paths.sort()
        # 将两路径配对成元组列表
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        # 打开RGB图像文件，并确保将其转换为RGB色彩模式
        img = Image.open(rgb_path).convert('RGB')
        # 对图像进一步预处理（基类定义的方法）
        img = self.rgb_transform(img)
        # 读取tiff文件，并调用read_tiff_organized_pc（）函数解析它并存储在organized_pc，其中包含了点云数据
        organized_pc = read_tiff_organized_pc(tiff_path)
        # organized_pc_to_depth_map（）函数将点云数据转换为深度图像（单通道），然后用np.repeat（）函数将其扩展为3通道的深度图
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        # 对三通道深度图进行调整大小
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        # 对原始的点云数据进行调整大小
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        # 创建副本并将其从计算图中分离，转换为浮点数类型
        resized_organized_pc = resized_organized_pc.clone().detach().float()
        # 返回一个元组，包含了图像、经过处理的有序点云数据及深度图，以及相应的标签
        return (img, resized_organized_pc, resized_depth_map_3channel), label

# 用于数据集加载和处理
class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path='datasets/eyecandies_preprocessed'):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        # 对标签图像进行处理
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        # 列出self.img_path下的不同缺陷类型（defect_types）
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            # 如果缺陷类别是'good',则加载RGB图像路径和tiff路径，并将他们组合成sample_paths,这些路径添加到img_tot_paths列表，同时添加了0到gt_tot_paths和tot_labels，表示这些样本是“好”的
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            # 如果缺陷类别不是'good',则加载RGB图像路径、tiff路径以及相应的gt图像路径，并将RGB路径和tiff路径组合成sample_paths,这些路径添加到img_tot_paths列表，同时添加了1到tot_labels，表示这些样本是“异常”的
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))
        # 确保img_tot_paths和gt_tot_paths的长度相同
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)
        # 从tiff文件中读取有序点云数据
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()
        

        
        # 标签信息是否为空
        if gt == 0:
            # 创建一个与深度图相同大小的全零张量作为标签
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            # 打开标签图像文件，将其转换为灰度模式
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            #将标签图像中大于0.5的像素值设为1，小于等于0.5的像素值设为0
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, rgb_path

# 批量加载数据
def get_data_loader(split, class_name, img_size, args):
    if split in ['train']:
        dataset = TrainDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)
    elif split in ['test']:
        dataset = TestDataset(class_name=class_name, img_size=img_size, dataset_path=args.dataset_path)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader
