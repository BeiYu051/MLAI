import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Unet import UNet3D
from load_data import MedicalDataset, transform
import torch

if __name__ == '__main__':
    # --------------------
    # load the index from the file
    index_file = 'dataset/trial.txt'
    indexs = []
    # read in the file and store the first colomn as the index
    # store the index start by 002
    with open(index_file, 'r') as f:
        for line in f:
            index = line.split()[0]
            if index[2] == '3':
                indexs.append(index)
    # --------------------
    print(len(indexs))
    # load the data
    image_paths = []
    label_paths = []
    for index in indexs:
        image_paths.append(f"dataset/data/{index}_img.nii")
        label_paths.append(f"dataset/data/{index}_mask.nii")

    dataset = MedicalDataset(image_paths, label_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    model = UNet3D(in_channels=8, out_channels=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # train the model
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    epochs = 10
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            # move the data to the device
            images = images.to(device)
            labels = labels.to(device)
            # print(labels.dtype, labels.shape)  # 确保为 long 类型且形状为 [batch_size]
            # print(labels.min(), labels.max())  # 确保标签值在 [0, num_classes - 1] 范围内
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            # print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
            outputs = model(images)
            # compute the loss
            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')

    # save the model
    torch.save(model.state_dict(), 'model.pth')

