from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from torch.nn import Module


class CLRDataset(Dataset):
    def __init__(
        self,
        name: str = "CIFAR10",
        root: str = "data/",
        train: bool = None,
        transforms: Module = transforms.ToTensor(),
    ):
        super().__init__()

        self.root = root
        self.train = train
        self.transforms = transforms
        self.dataset = getattr(datasets, name)(
            root=self.root, train=self.train, download=True
        )

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        image, _ = self.dataset[index]
        images = self.transforms(image)
        return images
    
    def __len__(self) -> int:
        return len(self.dataset)
