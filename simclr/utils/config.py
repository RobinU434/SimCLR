from dataclasses import dataclass

@dataclass
class _ResNetEncoder:
    base_model_name: str
    pretrained: bool


@dataclass
class _ProjectionHead:
    hidden_dim: int
    output_dim: int
    activation_func: str


@dataclass
class _Optimizer_params:
    lr: str


@dataclass
class _NTCrossEntropy:
    tau: int
    normalize: bool


@dataclass
class _Train_dataset:
    name: str
    root: str


@dataclass
class _Val_dataset:
    name: str
    root: str


@dataclass
class Config:
    ResNetEncoder: _ResNetEncoder
    projection_head_enabled: bool
    ProjectionHead: _ProjectionHead
    optimizer_name: str
    optimizer_params: _Optimizer_params
    NTCrossEntropy: _NTCrossEntropy
    train_dataset: _Train_dataset
    val_dataset: _Val_dataset
    max_epochs: int
    warmup_epochs: int
    val_check_interval: int
    precision: int
    batch_size: int
    device: str
    num_worker: int

    def __post_init__(self):
        self.ResNetEncoder = _ResNetEncoder(**self.ResNetEncoder)
        self.ProjectionHead = _ProjectionHead(**self.ProjectionHead)
        self.optimizer_params = _Optimizer_params(**self.optimizer_params)
        self.NTCrossEntropy = _NTCrossEntropy(**self.NTCrossEntropy)
        self.train_dataset = _Train_dataset(**self.train_dataset)
        self.val_dataset = _Val_dataset(**self.val_dataset)
