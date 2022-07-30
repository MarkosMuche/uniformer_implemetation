
import torch
from unifromer import Uniformer

model = Uniformer(
    num_classes = 2,                 # number of output classes
    dims = (64, 128, 256, 512),         # feature dimensions per stage (4 stages)
    depths = (3, 4, 8, 3),              # depth at each stage
    mhsa_types = ('l', 'l', 'g', 'g')   # aggregation type at each stage, 'l' stands for local, 'g' stands for global
)

import torch.nn as nn
import torch.optim as optim

from dataset import dataloader_train

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    dataloader_train_iter = iter(dataloader_train)
    for i, data in enumerate(dataloader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')