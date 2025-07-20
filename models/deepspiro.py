import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import logging

logger = logging.getLogger(__name__)

class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class MyConv1dPadSame(nn.Module):
    """1D convolution with 'SAME' padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        padded_x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.conv(padded_x)


class MyMaxPool1dPadSame(nn.Module):
    """1D max pooling with 'SAME' padding."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size  # In original code, stride is same as kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        padded_x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.max_pool(padded_x)


# --- Core Model Definitions (Corrected to match original state_dict) ---

class TemporalAttention(nn.Module):
    """Temporal Attention mechanism."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.swish = Swish()
        self.linear2 = nn.Linear(input_dim, 1, bias=False)
        self.bilinear_weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.bilinear_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        processed_x = self.swish(self.linear1(x))
        bilinear_out = torch.matmul(processed_x, self.bilinear_weight)
        logits = self.linear2(bilinear_out)
        return F.softmax(logits, dim=1)


class BasicBlock(nn.Module):
    """Basic residual block for Net1D. Structure restored to match original weights."""

    def __init__(self, in_channels: int, out_channels: int, ratio: float, kernel_size: int,
                 stride: int, groups: int, downsample: bool, is_first_block: bool = False,
                 use_bn: bool = True, use_do: bool = True):
        super().__init__()
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.use_do = use_do

        middle_channels = int(out_channels * ratio)

        # Layer definitions restored to match original structure
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, middle_channels, kernel_size=1, stride=1)

        self.bn2 = nn.BatchNorm1d(middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(middle_channels, middle_channels, kernel_size, self.stride, groups)

        self.bn3 = nn.BatchNorm1d(middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(middle_channels, out_channels, kernel_size=1, stride=1)

        r = 2
        self.se_fc1 = nn.Linear(out_channels, out_channels // r)
        self.se_fc2 = nn.Linear(out_channels // r, out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = x
        if not self.is_first_block:
            if self.use_bn: out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do: out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn: out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do: out = self.do2(out)
        out = self.conv2(out)

        if self.use_bn: out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do: out = self.do3(out)
        out = self.conv3(out)

        se = out.mean(-1)
        se = self.se_activation(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = torch.einsum('abc,ab->abc', out, se)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        out += identity
        return out


class BasicStage(nn.Module):
    """A stage of multiple BasicBlocks. Restored to use nn.ModuleList."""

    def __init__(self, in_channels: int, out_channels: int, ratio: float, kernel_size: int,
                 stride: int, groups: int, i_stage: int, m_blocks: int, use_bn: bool, use_do: bool, verbose: bool):
        super().__init__()
        self.block_list = nn.ModuleList()
        for i_block in range(m_blocks):
            is_first_block = i_stage == 0 and i_block == 0
            downsample = i_block == 0
            tmp_in_channels = in_channels if i_block == 0 else out_channels

            self.block_list.append(BasicBlock(
                in_channels=tmp_in_channels, out_channels=out_channels, ratio=ratio,
                kernel_size=kernel_size, stride=stride if downsample else 1, groups=groups,
                downsample=downsample, is_first_block=is_first_block, use_bn=use_bn, use_do=use_do
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.block_list:
            out = block(out)
        return out


class Net1D(nn.Module):
    """A 1D CNN for feature extraction. Restored to use nn.ModuleList."""

    def __init__(self, in_channels: int, base_filters: int, ratio: float, filter_list: list[int],
                 m_blocks_list: list[int], kernel_size: int, stride: int, groups_width: int,
                 use_bn: bool = True, use_do: bool = True, verbose: bool = False):
        super().__init__()
        self.first_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size, stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters) if use_bn else nn.Identity()
        self.first_activation = Swish()

        self.stage_list = nn.ModuleList()
        current_in_channels = base_filters
        for i, out_channels in enumerate(filter_list):
            m_blocks = m_blocks_list[i]
            groups = out_channels // groups_width
            self.stage_list.append(BasicStage(
                in_channels=current_in_channels, out_channels=out_channels, ratio=ratio,
                kernel_size=kernel_size, stride=stride, groups=groups, i_stage=i,
                m_blocks=m_blocks, use_bn=use_bn, use_do=use_do, verbose=verbose
            ))
            current_in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_activation(self.first_bn(self.first_conv(x)))
        for stage in self.stage_list:
            out = stage(out)
        return out


class DeepSpiro(nn.Module):
    """DeepSpiro model. Classifier name restored to 'dense'."""

    def __init__(self, in_channels: int, n_len_seg: int, n_classes: int, verbose: bool = False, **kwargs):
        super().__init__()
        self.n_len_seg = n_len_seg
        self.verbose = verbose

        self.cnn = Net1D(
            in_channels=in_channels, base_filters=8, ratio=1.0,
            filter_list=[16, 32, 32, 64], m_blocks_list=[2, 2, 2, 2],
            kernel_size=16, stride=2, groups_width=1, verbose=verbose,
            use_bn=True, use_do=True
        )
        self.out_channels_cnn = 64

        self.rnn = nn.LSTM(
            input_size=self.out_channels_cnn, hidden_size=self.out_channels_cnn,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.temporal_attention = TemporalAttention(input_dim=2 * self.out_channels_cnn)
        # ✨ --- KEY CHANGE: Renamed 'classifier' back to 'dense' --- ✨
        self.dense = nn.Linear(2 * self.out_channels_cnn, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_samples, n_segments, _, n_channel = x.shape
        lengths = mask.sum(dim=1).cpu().tolist()

        valid_segments_data = x[mask > 0]
        if n_channel != 1: valid_segments_data = valid_segments_data.unsqueeze(1)  # Ensure channel dim

        cnn_input = valid_segments_data.permute(0, 2, 1)  # (B*Seg, Channel, Len)

        cnn_output = self.cnn(cnn_input)
        cnn_output_pooled = cnn_output.mean(-1)

        reconstructed_output = torch.zeros(n_samples, n_segments, self.out_channels_cnn,
                                           device=cnn_output_pooled.device, dtype=cnn_output_pooled.dtype)

        valid_segment_idx = 0
        for i in range(n_samples):
            num_valid_segments = int(lengths[i])
            if num_valid_segments > 0:
                segments_for_sample = cnn_output_pooled[valid_segment_idx: valid_segment_idx + num_valid_segments]
                reconstructed_output[i, :num_valid_segments, :] = segments_for_sample
                valid_segment_idx += num_valid_segments

        # Ensure lengths are valid before packing
        valid_lengths = [l for l in lengths if l > 0]
        if not valid_lengths:
            # Handle case where all samples in batch are empty
            dummy_features = torch.zeros(n_samples, 1, 2 * self.out_channels_cnn, device=x.device, dtype=x.dtype)
            dummy_logits = torch.zeros(n_samples, self.dense.out_features, device=x.device, dtype=x.dtype)
            return dummy_features, dummy_logits

        packed_input = rnn_utils.pack_padded_sequence(reconstructed_output, lengths, batch_first=True,
                                                      enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        features, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        attention_weights = self.temporal_attention(features)
        context_vector = torch.sum(features * attention_weights, dim=1)

        logits = self.dense(context_vector)

        return features, logits