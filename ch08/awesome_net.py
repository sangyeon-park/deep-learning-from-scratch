# Create your awesome net!!
import torch
from deep_convnet import DeepConvNet

DCN = DeepConvNet()

input = torch.randn((3,1,28,28))
print(input)

DCN.predict(input)