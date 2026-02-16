"""Utility functions for model initialization."""

# Import packages
import torch.nn as nn
import torch.nn.init as init


def _init_rnn_weights(rnn: nn.RNNBase) -> None:
    """
    Initializes GRU/LSTM parameters:
      - weight_ih* -> Xavier uniform
      - weight_hh* -> Orthogonal
      - biases -> zeros (LSTM forget bias is set to 1)

    Parameters
    ----------
    rnn : nn.RNNBase
        The RNN module to initialize.
    """
    for name, p in rnn.named_parameters():
        if not p.requires_grad:
            continue
        if 'weight_ih' in name:
            init.xavier_uniform_(p.data)
        elif 'weight_hh' in name:
            # Orthogonal is a good default for recurrent connections
            init.orthogonal_(p.data)
        elif 'bias' in name:
            p.data.zero_()
            # For LSTM, set forget gate bias to 1
            if isinstance(rnn, nn.LSTM):
                # PyTorch layout for LSTM gate vectors: (i, f, g, o)
                hidden_size = p.data.shape[0] // 4
                # Set forget gate (slice [hidden_size:2*hidden_size]) to 1
                p.data[hidden_size : 2 * hidden_size].fill_(1.0)


def init_model_weights(module: nn.Module) -> None:
    """
    Walks through `module` and applies inits:
      - nn.Linear -> Xavier uniform (bias as zeros)
      - nn.Conv1d -> Xavier uniform (bias as zeros)
      - nn.LayerNorm -> weight=1, bias=0
      - nn.LSTM / nn.GRU -> _init_rnn_weights
    
    Example usage:
      model = EphysTokenizer(config)
      init_model_weights(model)

    Parameters
    ----------
    module : nn.Module
        The model or layer to initialize.
    """
    # If user passes the whole model, apply recursively
    for m in module.modules():
        # Linear
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        # Conv1d
        elif isinstance(m, nn.Conv1d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        # LayerNorm
        elif isinstance(m, nn.LayerNorm):
            if getattr(m, "weight", None) is not None:
                init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                init.zeros_(m.bias)
        # RNNs
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            _init_rnn_weights(m)
