import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    """
    MLP with a single hidden layer, implemented in PyTorch.

    Architecture: Re-create your MLP in PyTorch
        Input are 2 dimensions
          - Linear layer from input to hidden
          - Sigmoid activation
          - Linear layer towards output
          - Sigmoid activation (or might use softmax later-on)
          - Output (3)

    Notes:
    - The forward pass already returns sigmoid outputs (probabilities).
    - Loss function and optimizer are defined outside this class.
    - Backpropagation is handled automatically by PyTorch's autograd.
    """
    def __init__(self, hidden_dim=5):
        """
        Parameters:
            hidden_dim : int
                Number of neurons in the hidden layer.
        """
        super().__init__()

        # --------------------------------------------------
        # Layers
        # --------------------------------------------------
        # TODO: Create layers (as done in the exercises) and activation functions

        self.input_to_hidden = nn.Linear(in_features=2, out_features=hidden_dim)  # Input Layer to Hidden Layer
        
        self.sigmoid = nn.Sigmoid()                                               # Sigmoid activation - Hidden Layer
        
        self.reLU = nn.ReLU()                                                      # ReLU activation - Hidden Layer (if preferred)

        self.hidden_to_output = nn.Linear(in_features=hidden_dim, out_features=3) # Output Layer (Linear Layer towards output)
        
        self.softmax = nn.Softmax(dim=1)                                          # Softmax activation - Output Layer (if needed)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, x):
        """
        Forward pass of the network.

        x: Tensor of shape (N, 2)

        returns:
            y_hat: Tensor of shape (N, 3)
                   Sigmoid outputs (class-wise probabilities)
        """
        # TODO: Create forward pass
        # = instantiate the computation from input to output through the layers 
        # Calculate activations and apply activation functions afterwards.
        # Return the output (after the last activation function)

        # Initially, this will fail! There are no parameters and 
        # there is no working forward pass.
        # Pytorch will therefore fail creating autonmatically the backward pass / autograd.
        
        z1 = self.input_to_hidden(x)          # Linear transformation to hidden layer
        a1 = self.sigmoid(z1)                 # Sigmoid activation
        
        z2 = self.hidden_to_output(a1)        # Linear transformation to output layer
        
        y_hat_sigmoid = self.sigmoid(z2)      # Sigmoid activation
        y_hat_softmaxed = self.softmax(z2)    # Softmax activation (if preferred)


        return y_hat_sigmoid                  # or return y_hat_softmaxed if softmax is preferred