
import torch
from torch import nn

class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Insertar código
        self.weight = torch.randn(output_size, input_size)
        self.bias = torch.zeros(output_size)

    def forward(self, x):
        # Insertar código
        return torch.mm(x, self.weight.t()) + self.bias

class CustomSigmoid(nn.Module):
    def forward(self, x):
        # Insertar código
        return 1 / (1 + torch.exp(-x))

class CustomSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        # Insertar código
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        # Insertar código
        for layer in self.layers:
            x = layer(x)
        return x
out=CustomLinear(3,5)
out2=CustomSigmoid()
x = torch.tensor([[0, 1, 2]]).float()
print(out(x))
print(out2(x))
capa_secuencial=CustomSequential(out,out2)
print(capa_secuencial)