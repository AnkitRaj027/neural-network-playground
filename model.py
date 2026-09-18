import torch.nn as nn
import torch
class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation


    ):
        super().__init__()

        layers=[]
        current_size=input_size
        for hidden_size in hidden_sizes:
            layers.append(
                nn.Linear(current_size, hidden_size)
            )
            layers.append(
                activation()
            )
            current_size=hidden_size
        layers.append(
            nn.Linear(current_size, output_size)
        )
        self.network=nn.Sequential(*layers)
        
    def forward(self,x):
        return self.network(x)
        

# model= NeuralNetwork()
# X=torch.randn(5,2)
# output=model(X)
# print(f"{X.shape}\n {output.shape}\n {output}")