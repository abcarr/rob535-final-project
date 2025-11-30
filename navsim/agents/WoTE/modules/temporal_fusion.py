"""
Temporal BEV Fusion Module using ConvGRU

Implements multi-frame BEV aggregation with ego-motion compensation
for improved motion understanding in autonomous driving.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    """Convolutional GRU Cell for spatial-temporal feature processing."""
    
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvGRU cell.
        
        Args:
            input_size: (int, int) - Height and width of input tensor as (height, width)
            input_dim: int - Number of channels of input tensor
            hidden_dim: int - Number of channels of hidden state
            kernel_size: (int, int) - Size of the convolutional kernel
            bias: bool - Whether or not to add the bias
            dtype: torch.cuda.FloatTensor or torch.FloatTensor - Device type
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * self.hidden_dim,  # for update_gate, reset_gate
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        self.conv_can = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def init_hidden(self, batch_size):
        """Initialize hidden state with zeros."""
        return Variable(
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        ).type(self.dtype)

    def forward(self, input_tensor, h_cur):
        """
        Forward pass through ConvGRU cell.
        
        Args:
            input_tensor: (b, c, h, w) - Current input
            h_cur: (b, c_hidden, h, w) - Current hidden state
            
        Returns:
            h_next: (b, c_hidden, h, w) - Next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU(nn.Module):
    """Multi-layer Convolutional GRU for temporal sequence processing."""
    
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        dtype,
        batch_first=False,
        bias=True,
        return_all_layers=False
    ):
        """
        Initialize ConvGRU.
        
        Args:
            input_size: (int, int) - Height and width of input
            input_dim: int - Number of channels of input tensor
            hidden_dim: int or list - Number of channels of hidden state
            kernel_size: (int, int) - Size of convolutional kernel
            num_layers: int - Number of ConvGRU layers
            dtype: torch dtype - Device type
            batch_first: bool - If True, input shape is (B, T, C, H, W)
            bias: bool - Whether to add bias
            return_all_layers: bool - Return outputs from all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure kernel_size and hidden_dim are lists
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(
                ConvGRUCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    dtype=self.dtype
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass through ConvGRU.
        
        Args:
            input_tensor: (B, T, C, H, W) or (T, B, C, H, W)
            hidden_state: Optional previous hidden states
            
        Returns:
            layer_output_list: List of outputs from each layer
            last_state_list: List of final hidden states
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError("Stateful ConvGRU not implemented yet")
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    h_cur=h
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """Convert single value to list if needed."""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class TemporalBEVFusion(nn.Module):
    """
    Temporal BEV Fusion using ConvGRU with ego-motion compensation.
    
    Aggregates multi-frame BEV features through:
    1. Ego-motion compensation (spatial alignment via affine warping)
    2. Temporal aggregation (ConvGRU recurrent updates)
    """
    
    def __init__(
        self,
        bev_channels=512,
        hidden_dim=512,
        bev_size=(8, 8),
        num_history=4,
        kernel_size=(3, 3),
        num_layers=1
    ):
        """
        Initialize TemporalBEVFusion.
        
        Args:
            bev_channels: int - Number of channels in BEV features
            hidden_dim: int - Hidden dimension for ConvGRU
            bev_size: (int, int) - Spatial size of BEV (H, W)
            num_history: int - Number of historical frames
            kernel_size: (int, int) - Convolution kernel size
            num_layers: int - Number of ConvGRU layers
        """
        super().__init__()
        
        self.bev_size = bev_size
        self.num_history = num_history
        self.bev_channels = bev_channels
        
        # ConvGRU for temporal fusion
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        self.conv_gru = ConvGRU(
            input_size=bev_size,
            input_dim=bev_channels,
            hidden_dim=[hidden_dim] * num_layers,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dtype=dtype,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
    def warp_bev_features(self, bev_features, ego_motion):
        """
        Warp historical BEV frames to align with current frame using ego-motion.
        
        Args:
            bev_features: (B, T, C, H, W) - Multi-frame BEV features
            ego_motion: (B, T-1, 2, 3) - Affine transformation matrices
            
        Returns:
            warped_features: (B, T, C, H, W) - Aligned BEV features
        """
        B, T, C, H, W = bev_features.shape
        warped = []
        
        # Current frame (t=T-1) doesn't need warping - use as reference
        warped.append(bev_features[:, -1])
        
        # Warp historical frames (t=0 to T-2) to align with current frame
        for t in range(T - 1):
            prev_bev = bev_features[:, t]  # (B, C, H, W)
            affine = ego_motion[:, t]       # (B, 2, 3)
            
            # Create sampling grid for spatial transformer
            grid = F.affine_grid(
                affine,
                prev_bev.size(),
                align_corners=False
            )
            
            # Warp using bilinear interpolation
            warped_bev = F.grid_sample(
                prev_bev,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
            warped.insert(0, warped_bev)
        
        return torch.stack(warped, dim=1)  # (B, T, C, H, W)
    
    def forward(self, bev_features, ego_motion, hidden_state=None):
        """
        Apply temporal BEV fusion with ego-motion compensation.
        
        Args:
            bev_features: (B, T, C, H, W) - Multi-frame BEV features
            ego_motion: (B, T-1, 2, 3) - Ego-motion affine transformations
            hidden_state: Optional previous hidden state for stateful operation
            
        Returns:
            fused_bev: (B, C, H, W) - Temporally fused BEV feature
            hidden_states: Hidden states for next iteration (if stateful)
        """
        # Step 1: Warp historical BEVs to align with current frame
        aligned_bevs = self.warp_bev_features(bev_features, ego_motion)
        
        # Step 2: Apply ConvGRU for temporal aggregation
        layer_output_list, hidden_states = self.conv_gru(aligned_bevs, hidden_state)
        
        # Step 3: Extract fused BEV from last timestep of output sequence
        fused_bev = layer_output_list[0][:, -1]  # (B, C, H, W)
        
        return fused_bev, hidden_states
