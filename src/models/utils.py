import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        activation: Type[nn.Module] = nn.SiLU,
        weight_init: str = 'xavier',
        type: str = 'trm'
    ):
        super().__init__()
        
        assert depth >= 1, "Depth must be at least 1"
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.activation = activation()
        
        layers = []
        if depth == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(depth - 2):
                layers.append(self.activation)
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.initialize_weights(weight_init)
        
        
    def initialize_weights(self, method: str):
        for m in self.network:
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise ValueError(f"Unknown weight initialization method: {method}")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, 
                a: torch.Tensor,
                b: torch.Tensor,
                c: Optional[torch.Tensor] = None,
                type: str = 'trm'
                ) -> torch.Tensor:
        if type == 'trm':            
            if c is None:
                c = torch.zeros_like(a)
            
            x = torch.cat([a, b, c], dim = -1)
            h = self.activation(self.network[0](x))
            out = self.network[1:](h)
        elif type == 'transformer':
            # standard MLP forward pass
            x = torch.cat([a, b], dim = -1)
            out = self.network(x)
        else:
            raise ValueError(f"Unknown block type: {type}")
        return out       
    
class OutputHead(nn.Module):
    """
    Output head for projecting latent state to output.
    """
    def __init__(self, 
                 dim:int , num_classes:int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, y):
        """
        Forward pass of the output head.
        Args:
            y (torch.Tensor): Latent state tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.linear(y)
class EMA:
    """
    Exponential Moving Average for model parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update the EMA parameters with the current model parameters.
        Args:
            model (nn.Module): The model to update from.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(param.data * (1.0 - self.decay))

    def apply(self, model: nn.Module):
        """
        Apply the EMA parameters to the model.
        Args:
            model (nn.Module): The model to apply the EMA parameters to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
                