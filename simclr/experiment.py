import logging
from lightning import Trainer
from simclr.models.lightning_model import SimCLRLightning
from simclr.utils.config import Config
from simclr.utils.filesystem import load_yaml
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from simclr.dataset.dataset import CLRDataset
from simclr.dataset.augmentation import ContrastiveTransformations, contrast_transforms


class Experiment:
    """Start experiment with the SimCLR framework"""

    def __init__(self):
        """init instance of Experiment class"""
        pass

    def train(
        self, save_path: str = None, config_path: str = "./configs/simclr_config.yaml"
    ):
        """train a SimCLR model

        Args:
            save_path (str, optional): where to store checkpoints and save train metrics. Defaults to None.
            config_path (str, optional): from where to load the config. Defaults to "./configs/simclr_config.yaml".
        """
        config_yaml = load_yaml(config_path)
        config = Config(**config_yaml)

        if not torch.cuda.is_available() and "cuda" in config.device:
            logging.warning("No cuda device available. Fall back to cpu")
            device = "cpu"
        else:
            device = config.device
        device = torch.device(device)

        if save_path is None:
            save_path = "./results/"

        train_loader_config = {
            **config.train_dataset.__dict__,
            "train": True,
            "transforms": ContrastiveTransformations(
                base_transforms=contrast_transforms,
                n_views=2,
            ),
        }
        dataset = CLRDataset(**train_loader_config)
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_worker,
        )

        val_loader_config = {
            **config.val_dataset.__dict__,
            "train": False,
            "transforms": ContrastiveTransformations(
                base_transforms=contrast_transforms,
                n_views=2,
            ),
        }
        dataset = CLRDataset(**val_loader_config)
        val_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_worker,
        )
        
        model = SimCLRLightning(
            config_yaml,
            device=device,
            input_shape=dataset[0][0][None].shape,
            T_max=len(train_loader),
        )
        trainer = Trainer(
            precision=config.precision,
            logger=[
                CSVLogger(save_dir=save_path),
                TensorBoardLogger(save_dir=save_path),
            ],
            max_epochs=config.max_epochs,
            val_check_interval=config.val_check_interval,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
