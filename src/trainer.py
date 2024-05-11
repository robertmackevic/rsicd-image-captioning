from torch.utils.data.dataloader import DataLoader

from src.utils import get_available_device, get_logger, load_config


class Trainer:
    def __init__(self, train_dl: DataLoader, val_dl: DataLoader) -> None:
        self.config = load_config()
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.summary_writer_train = None
        self.summary_writer_eval = None
