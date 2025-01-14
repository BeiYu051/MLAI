import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt

from Unet import UNet3D
from load_data import MedicalDataset, transform

batch_size = 32

if __name__ == '__main__':
    # --------------------
    # load the index from the file
    index_file = 'dataset/trial.txt'
    indexs = []
    # read in the file and store the first colomn as the index
    # store the index start by 003
    # remove the last ones which cannot mod 8
    with open(index_file, 'r') as f:
        for line in f:
            index = line.split()[0]
            if index[2] == '3':
                indexs.append(index)
    # remove the last ones which cannot mod 8
    indexs = indexs[:-(len(indexs) % batch_size)]
    # --------------------
    print(len(indexs))
    # load the data
    image_paths = []
    label_paths = []
    for index in indexs:
        image_paths.append(f"dataset/data/{index}_img.nii")
        label_paths.append(f"dataset/data/{index}_mask.nii")

    dataset = MedicalDataset(image_paths, label_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    model = UNet3D(in_channels=1, out_channels=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # train the model
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    epochs = 10
    # plot the loss
    history = {'train': []}
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            # move the data to the device
            images = images.to(device)
            labels = labels.to(device)
            # print(f"Images shape: {images.shape}")  # 应为 [batch_size, depth, height, width]
            # print(f"Labels shape: {labels.shape}")
            # print(labels.min(), labels.max())  # 确保标签值在 [0, num_classes - 1] 范围内
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            # print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
            outputs = model(images)
            # compute the loss
            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            loss = criterion(outputs, labels)
            # store the loss
            history['train'].append(loss.item())
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()
            print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')
    # plot the loss
    plt.plot(history['train'], label='train')
    # save the model
    torch.save(model.state_dict(), 'model.pth')

