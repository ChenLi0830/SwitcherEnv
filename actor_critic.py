import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP


class RecurrentActorProb(nn.Module):
  """Recurrent version of ActorProb.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

  def __init__(
      self,
      layer_num: int,
      state_shape: Sequence[int],
      action_shape: Sequence[int],
      hidden_layer_size: int = 128,
      device: Union[str, int, torch.device] = "cpu",
  ) -> None:
    super().__init__()
    self.device = device
    self.nn = nn.LSTM(
        input_size=int(np.prod(state_shape)),
        hidden_size=hidden_layer_size,
        num_layers=layer_num,
        batch_first=True,
    )
    output_dim = int(np.prod(action_shape))
    self.softmax_output = softmax_output

  def forward(
      self,
      obs: Union[np.ndarray, torch.Tensor],
      state: Optional[Dict[str, torch.Tensor]] = None,
      info: Dict[str, Any] = {},
  ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
    """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
    obs = torch.as_tensor(
        obs,
        device=self.device,
        dtype=torch.float32,
    )
    # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
    # In short, the tensor's shape in training phase is longer than which
    # in evaluation phase.
    if len(obs.shape) == 2:
      obs = obs.unsqueeze(-2)
    self.nn.flatten_parameters()
    if state is None:
      obs, (hidden, cell) = self.nn(obs)
    else:
      # we store the stack data in [bsz, len, ...] format
      # but pytorch rnn needs [len, bsz, ...]
      obs, (hidden, cell) = self.nn(obs, (state["hidden"].transpose(
          0, 1).contiguous(), state["cell"].transpose(0, 1).contiguous()))
    logits = obs[:, -1]
    # please ensure the first dim is batch size: [bsz, len, ...]
    return , {
        "hidden": hidden.transpose(0, 1).detach(),
        "cell": cell.transpose(0, 1).detach()
    }


class RecurrentCritic(nn.Module):
  """Recurrent version of Critic.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

  def __init__(
      self,
      layer_num: int,
      state_shape: Sequence[int],
      action_shape: Sequence[int] = [0],
      device: Union[str, int, torch.device] = "cpu",
      hidden_layer_size: int = 128,
  ) -> None:
    super().__init__()
    self.state_shape = state_shape
    self.action_shape = action_shape
    self.device = device
    self.nn = nn.LSTM(
        input_size=int(np.prod(state_shape)),
        hidden_size=hidden_layer_size,
        num_layers=layer_num,
        batch_first=True,
    )
    self.fc2 = nn.Linear(hidden_layer_size + int(np.prod(action_shape)), 1)

  def forward(
      self,
      obs: Union[np.ndarray, torch.Tensor],
      act: Optional[Union[np.ndarray, torch.Tensor]] = None,
      info: Dict[str, Any] = {},
  ) -> torch.Tensor:
    """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
    obs = torch.as_tensor(
        obs,
        device=self.device,
        dtype=torch.float32,
    )
    # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
    # In short, the tensor's shape in training phase is longer than which
    # in evaluation phase.
    assert len(obs.shape) == 3
    self.nn.flatten_parameters()
    obs, (hidden, cell) = self.nn(obs)
    obs = obs[:, -1]
    if act is not None:
      act = torch.as_tensor(
          act,
          device=self.device,
          dtype=torch.float32,
      )
      obs = torch.cat([obs, act], dim=1)
    obs = self.fc2(obs)
    return obs
