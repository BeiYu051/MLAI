import nibabel as nib
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

def resample_image(image, target_shape):
    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    target_shape = [1,1] + list(target_shape)
    img_tensor = F.interpolate(img_tensor, size=target_shape[2:], mode='trilinear', align_corners=False)
    return img_tensor.squeeze(0).squeeze(0).numpy()

def load_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    # resample the image to the target shape
    target_shape = (180, 180, 64)
    data = resample_image(data, target_shape)
    tensor = torch.tensor(data, dtype=torch.float32)
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
        # print(self.image_paths[idx])
        image = load_nii(self.image_paths[idx])
        # print(self.label_paths[idx])
        label = load_nii(self.label_paths[idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
