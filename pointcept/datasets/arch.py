import os
from .defaults import DefaultDataset
from .builder import DATASETS

@DATASETS.register_module()
class ArchDataset(DefaultDataset):
    """Arch Dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)