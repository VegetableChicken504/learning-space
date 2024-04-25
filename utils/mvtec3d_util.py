import tifffile as tiff
import torch


"""
有组织的点云：通常以一个矩阵或三维数组的形式表示，其中第一个维度对应于行，第二个维度对应于列。第三个维度包含点云的属性（例如x、y、z坐标）
无组织的点云：一个简单的二维数组，其中每一行代表一个点（包含点的坐标信息xyz），每列代表点的不同属性
"""
# 将有组织的点云数据转换为无组织的点云数据
def organized_pc_to_unorganized_pc(organized_pc):
    # 将有组织点云的行和列合并在一起，得到无组织的点云数据
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

# 读取tiff格式图像并返回其内容的简单函数
def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img

# 调整有组织点云大小的函数
def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    # 先将点云转换为张量，用permute()对维度进行置换，以确保维度的顺序是正确的（通常是通道数、高度、宽度）；接着用unsqueeze()在第0维度（batch维度）上增加一个维度，将数据转换为batch的形式；最后contiguous()确保张量是连续的
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    # 用插值操作调整点云大小
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        # 返回张量
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        # 返回数组
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()

# 从有组织的点云数据中提取深度图像
def organized_pc_to_depth_map(organized_pc):
    # 使用numpy切片操作从点云中选择所有行和列的第三个属性，即z坐标。这将生成一个与输入点云数据的行数列数相同的矩阵，其中每个元素代表相应点的深度值
    return organized_pc[:, :, 2]
