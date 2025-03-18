from torch import nn
import os

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=4)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,256, kernel_size=3, padding=1, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256,512, kernel_size=3, padding=1, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(512)

        # Add more layers...
        self.linear_relu_stack= nn.Sequential( #ongi strriude diviso 2
            nn.Linear(14*14*512 , 1024),
            nn.ReLU(),
            nn.Linear(1024 , 512),
            nn.ReLU(),
            nn.Linear(512 , 256),
            nn.ReLU(),
             nn.Linear(256 , 200)
           )
        # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x).relu()

        x = self.conv2(x)
        x = self.batch_norm2(x).relu()

        x = self.conv3(x)
        x = self.batch_norm3(x).relu()

        x = self.conv4(x)
        x = self.batch_norm4(x).relu()

        x = self.flatten(x)
        x = self.linear_relu_stack(x)

        return x
    

model = CustomNet().to(device)
