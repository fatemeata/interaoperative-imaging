from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.cuda
import network
import dataset

if __name__ == "__main__":

    wrist_model = network.CNN()
    data_path = "data"
    train_loader = DataLoader(dataset.WristDataset(data_path), batch_size=8)
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10)

    trainer.fit(wrist_model, train_loader)

