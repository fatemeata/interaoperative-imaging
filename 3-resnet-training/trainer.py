from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.cuda
import wrist_network
import dataset
from utils.data_splitting import DataSplitting
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":

    # 1. init dataset
    root_dir = "data"
    wrist_dataset = dataset.WristDataset(root_dir=root_dir)

    # 2. split dataset
    train, validation, test = DataSplitting().k_fold_data(data=wrist_dataset)

    # 3. instantiate dataloader for train, valid, test dataset
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(validation, batch_size=1, shuffle=False, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

    # 4. create trainer
    wrist_model = wrist_network.WristResNetwork()

    logger = TensorBoardLogger(save_dir="loss_logs", name="wrist_model")

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=50,
        log_every_n_steps=30,
        logger=logger
    )

    # 5. use trainer to optimize the model, given training and validation dataloader
    trainer.fit(wrist_model, train_dataloader, valid_dataloader)
    trainer.test(dataloaders=test_dataloader, ckpt_path="best")

