from torchvision.datasets import SVHN

from .cifar10_datamodule import CIFAR10DataModule
from .utils import per_class_random_split

# for type hint
from typing import Optional, List
from torch.utils.data import Dataset


class SVHNDataModule(CIFAR10DataModule):
    num_classes: int = 10

    total_train_size: int = 73_257
    total_test_size: int = 26_032

    DATASET = SVHN

    def __init__(self,
                 data_dir: str,
                 labeled_train_size: int,
                 validation_size: int,
                 unlabeled_train_size: Optional[int] = None,
                 **kwargs):
        super(SVHNDataModule, self).__init__(
            data_dir=data_dir,
            labeled_train_size=labeled_train_size,
            validation_size=validation_size,
            unlabeled_train_size=unlabeled_train_size,
            **kwargs)

        # dataset stats
        # SVHN mean, std values in CHW
        self.dataset_mean = [0.4376821, 0.4437697, 0.47280442]
        self.dataset_std = [0.19803012, 0.20101562, 0.19703614]

    def prepare_data(self, *args, **kwargs):
        self.DATASET(root=self.data_dir, split="train", download=True)
        self.DATASET(root=self.data_dir, split="test", download=True)

    def setup(self, stage: Optional[str] = None):
        full_train_set = self.DATASET(root=self.data_dir, split="train")
        full_test_set = self.DATASET(root=self.data_dir, split="test")

        self.setup_helper(full_train_set=full_train_set, full_test_set=full_test_set, stage=stage)

    def split_dataset(self, dataset: Dataset, **kwargs) -> List[Dataset]:
        split_kwargs = dict(lengths=[self.validation_size, self.labeled_train_size],
                            num_classes=self.num_classes,
                            uneven_split=True)

        # update split arguments
        split_kwargs.update(kwargs)

        return per_class_random_split(dataset, **split_kwargs)
