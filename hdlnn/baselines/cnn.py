import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from hdlnn.contracts.interfaces import ISequenceModel

class CNN1DBaseline(nn.Module, ISequenceModel):
    """Causal 1D Convolutional Neural Network baseline model.
    
    Processes temporal sequence data using causal padding. Encodes sequence history 
    directly within the trajectory state tensor to maintain statelessness in streaming mode.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, proj_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.proj_dim = proj_dim
        self.padding = kernel_size - 1
        
        # 1D Causal Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size
        )
        
        # Optional projection head to project back to hypervector space
        if proj_dim:
            self.projection_head = nn.Linear(hidden_dim, proj_dim)
        else:
            self.projection_head = None

    def forward(
        self, 
        x: torch.Tensor, 
        h0: Optional[torch.Tensor] = None, 
        dt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates causal 1D convolution over sequence inputs.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            h0: Unused placeholder for compatibility.
            dt: Unused placeholder for compatibility.
        """
        batch_size, seq_len, _ = x.shape
        
        # Conv1d expects shape [batch_size, input_dim, seq_len]
        x_trans = x.transpose(1, 2)
        
        # Causal padding on the left of the sequence dimension
        padded_x = F.pad(x_trans, (self.padding, 0))
        
        # Conv output shape: [batch_size, hidden_dim, seq_len]
        conv_out = self.conv(padded_x)
        
        # Transpose back to [batch_size, seq_len, hidden_dim]
        outputs = conv_out.transpose(1, 2)
        
        # Retrieve final step output
        h_last = outputs[:, -1, :]
        
        return outputs, h_last

    def step(self, x: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Evaluates a single step using causal convolution.
        
        Encodes the sliding history of inputs inside h_prev to maintain compatibility.
        h_prev has shape [batch_size, (kernel_size - 1) * input_dim + hidden_dim].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
            
        batch_size = x.size(0)
        history_len = self.kernel_size - 1
        history_dim = history_len * self.input_dim
        
        # 1. Parse h_prev into input history and cnn output
        # If h_prev is zero/empty, initialize history to zeros
        if torch.all(h_prev == 0.0) and h_prev.size(1) == self.hidden_dim:
            history = torch.zeros((batch_size, history_dim), device=x.device, dtype=x.dtype)
        else:
            history = h_prev[:, :history_dim]
            
        # Reshape history to [batch_size, input_dim, history_len]
        history_tensor = history.view(batch_size, history_len, self.input_dim).transpose(1, 2)
        
        # 2. Append current input x to history
        # x is [batch_size, input_dim] -> unsqueeze to [batch_size, input_dim, 1]
        current_x = x.unsqueeze(2)
        combined_inputs = torch.cat([history_tensor, current_x], dim=2)  # [batch_size, input_dim, kernel_size]
        
        # 3. Run Conv1D
        conv_out = self.conv(combined_inputs)  # [batch_size, hidden_dim, 1]
        cnn_output = conv_out.squeeze(2)  # [batch_size, hidden_dim]
        
        # 4. Slide history window
        # Drop the oldest step, keep the rest + current_x
        new_history_tensor = combined_inputs[:, :, 1:]  # [batch_size, input_dim, history_len]
        new_history = new_history_tensor.transpose(1, 2).contiguous().view(batch_size, history_dim)
        
        # 5. Package new history + new cnn_output as next state
        h_next = torch.cat([new_history, cnn_output], dim=1)
        return h_next

    def predict_next_vector(self, state: torch.Tensor) -> torch.Tensor:
        """Projects the CNN hidden state back to hypervector space (D)."""
        if self.projection_head is None:
            raise ValueError("Projection head not configured in CNN1DBaseline.")
            
        # If state contains concatenated history, slice the CNN output portion (last hidden_dim elements)
        state_dim = state.size(-1)
        expected_output_dim = self.hidden_dim
        if state_dim > expected_output_dim:
            state = state[:, -expected_output_dim:]
            
        return self.projection_head(state)
