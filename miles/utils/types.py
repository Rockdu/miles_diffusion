from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import torch
import base64
from safetensors.torch import load, save

class LazyTensor(ABC):
    """Deferred load: **base64 ``str``** matching :func:`tensor_to_base64`.

    Public API: :meth:`resolve` → **CPU** :class:`torch.Tensor` (idempotent: decodes at most once, then returns cache).
    """

    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def resolve(self) -> torch.Tensor:
        """Materialize to a CPU tensor; repeated calls return the same tensor."""
        if self.tensor is None:
            self.tensor = self._resolve_tensor()
        return self.tensor
    
    @abstractmethod
    def _resolve_tensor(self) -> torch.Tensor:
        raise NotImplementedError

class LazyTensorFromTensor(LazyTensor):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.tensor = tensor

    def _resolve_tensor(self) -> torch.Tensor:
        return self.tensor

class SafetensorsBase64LazyTensor(LazyTensor):
    """Lazy tensor from rollout wire: base64-encoded safetensors blob (default tensor key ``\"t\"`` in :func:`~miles.utils.diffusion_rollout_response.decode_tensor_base64`)."""

    def __init__(self, b64: str) -> None:
        super().__init__()
        self.b64: str | None = b64

    def _resolve_tensor(self) -> torch.Tensor:
        if not self.b64:
            raise ValueError("SafetensorsBase64LazyTensor: b64 must be a non-empty str")
        return decode_tensor_base64(self.b64).detach().cpu()

def decode_tensor_base64(b64: str) -> torch.Tensor:
    """Deserialize base64 to CPU tensor (same wire format as inference: safetensors ``[\"t\"]``, else ``torch.load``)."""
    raw = base64.b64decode(b64.encode("ascii") if isinstance(b64, str) else b64)
    return load(raw)["t"]

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Encode a CPU tensor as base64 safetensors (single key ``tensor_key``, default ``t``)."""
    tensor = tensor.detach().cpu()
    raw = save({"t": tensor})
    return base64.b64encode(raw).decode("ascii")


def as_lazy_tensor(value: Any) -> LazyTensor | None:
    """If ``value`` is a base64 string from JSON, wrap as lazy tensor; else pass through (HTTP bodies never contain ``torch.Tensor``)."""
    if value is None:
        return None
    if isinstance(value, str):
        return SafetensorsBase64LazyTensor(b64=value)
    raise TypeError(f"Cannot convert {type(value)} to LazyTensor")

# Tensor field: either deferred safetensors+b64 or already materialized (e.g. after ``resolve()``).
RolloutTensorRef = LazyTensor

@dataclass
class RolloutDebugTensors:
    rollout_variance_noises: RolloutTensorRef | None = None
    rollout_prev_sample_means: RolloutTensorRef | None = None
    rollout_noise_std_devs: RolloutTensorRef | None = None
    rollout_model_outputs: RolloutTensorRef | None = None


@dataclass
class CondKwargs:
    txt_seq_lens: list[int] | None = None
    freqs_cis: list[RolloutTensorRef] | None = None
    img_shapes: list[list[tuple[int, int, int]]] | None = None
    encoder_hidden_states: list[RolloutTensorRef] | None = None


@dataclass
class DenoisingEnv:
    image_kwargs: Any | None = None
    pos_cond_kwargs: CondKwargs | None = None
    neg_cond_kwargs: CondKwargs | None = None
    guidance: Any | None = None


@dataclass
class DiTTrajectory:
    latent_model_inputs: RolloutTensorRef | None = None
    timesteps: RolloutTensorRef | None = None


@dataclass
class Sample:
    """The sample generated.

    Diffusion image rollout: fill from sglang-diffusion ``POST /rollout/generate`` via
    `apply_rollout_image_response`
    """

    group_index: int | None = None
    index: int | None = None
    # correlation id from rollout engine (e.g. UUID string)
    request_id: str | None = None
    # prompt
    prompt: str = ""
    # reproducibility
    seed: int | None = None
    # Lazy: :class:`SafetensorsBase64LazyTensor`; eager: :class:`torch.Tensor`. Image rollout: ``[C, T, H, W]`` (``T==1`` typical).
    generated_output: RolloutTensorRef | None = None
    rollout_log_probs: RolloutTensorRef | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: DenoisingEnv | None = None
    dit_trajectory: DiTTrajectory | None = None

    inference_time_s: float | None = None
    peak_memory_mb: float | None = None

    reward: dict[str, Any] | None = None
    weight_versions: list[str] = field(default_factory=list)

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]
