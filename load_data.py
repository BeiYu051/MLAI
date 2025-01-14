import nibabel as nib
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

def resample_image(image, target_shape):
    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    target_shape = [1,1] + list(target_shape)
    img_tensor = F.interpolate(img_tensor, size=target_shape[2:], mode='trilinear', align_corners=False)
    return img_tensor.squeeze(0).squeeze(0).numpy()

def resample_label(label, target_shape):
    label = torch.tensor(label, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    label_resampled = F.interpolate(label, size=target_shape, mode='nearest')
    label_resampled = label_resampled.squeeze().float().numpy()
    # set the label to 0,...,8
    label_resampled = np.round(label_resampled)
    return label_resampled

def load_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)
    # resample the image to the target shape
    target_shape = (180, 180, 16)
    data = resample_image(data, target_shape)
    # standardize the image
    data = (data - data.mean()) / data.std()
    tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # print(tensor.shape)
    return tensor

transform = Compose([
    Normalize(mean=[0.5], std=[0.5]),
])

class MedicalDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_nii(self.image_paths[idx])
        label = nib.load(self.label_paths[idx]).get_fdata()
        # pad the label to the target shape
        target_shape = (180, 180, 16)
        label = resample_label(label, target_shape)
        label = torch.tensor(label, dtype=torch.float32).permute(2, 0, 1)
        # label = load_nii(self.label_paths[idx])
        if self.transform:
            image = self.transform(image)
            # label = self.transform(label)
        # print(image.shape, label.shape)
        return image, label
