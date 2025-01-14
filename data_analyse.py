import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

#定义路径
image_path = "dataset/data/001000_img.nii"  # 替换为实际路径
mask_path = "dataset/data/001000_mask.nii"  # 替换为实际路径

# 使用 nibabel 加载 NIfTI 文件
image_nii = nib.load(image_path)
mask_nii = nib.load(mask_path)

# 提取数据
image_data = image_nii.get_fdata()
mask_data = mask_nii.get_fdata()

# 打印基本信息
print("Image shape:", image_data.shape)
print("Mask shape:", mask_data.shape)
print("Image dtype:", image_data.dtype)
print("Mask dtype:", mask_data.dtype)

# 查看掩膜的唯一值
unique_labels = np.unique(mask_data)
print("Unique labels in the mask:", unique_labels)

# 提取膀胱区域的掩膜
bladder_mask = (mask_data == 1).astype(np.float32)

# 检查膀胱掩膜的形状和体素数量
print("Bladder mask shape:", bladder_mask.shape)
print("Number of bladder voxels:", np.sum(bladder_mask))

def visualize_bladder(image, mask):
    """
    可视化膀胱区域叠加在原始图像上。
    Args:
        image (np.ndarray): 图像数据。
        mask (np.ndarray): 掩膜数据。
    """
    # 找到包含膀胱区域的切片索引
    nonzero_slices = np.where(np.any(mask > 0, axis=(0, 1)))[0]

    for slice_index in nonzero_slices:
        plt.figure(figsize=(8, 8))
        plt.imshow(image[:, :, slice_index], cmap="gray")  # 显示原始图像
        plt.imshow(mask[:, :, slice_index], cmap="hot", alpha=0.5)  # 膀胱区域叠加
        plt.title(f"Slice {slice_index} with Bladder Region")
        plt.axis("off")
        plt.show()

# 可视化膀胱区域
visualize_bladder(image_data, bladder_mask)