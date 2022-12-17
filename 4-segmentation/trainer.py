from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.cuda

import network
from utils.unet import UNet
import dataset
from utils.data_splitting import DataSplitting
from pytorch_lightning.loggers import TensorBoardLogger
from utils.data_augmentation import RadiusDataAugmentation, WristLineDataAugmentation

if __name__ == "__main__":

    # 1. init dataset
    root_dir = "data"
    radius_dataset = dataset.RadiusPointsDataset(root_dir=root_dir, transform=RadiusDataAugmentation())

    # 2. split dataset
    train, validation, test = DataSplitting().k_fold_data(data=radius_dataset)

    # 3. instantiate dataloader for train, valid, test dataset
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(validation, batch_size=4, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

    # 4. create trainer
    model = network.RadiusNetwork()
    
    logger = TensorBoardLogger(save_dir="loss_logs", name="wrist_model")

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=100,
        log_every_n_steps=30,
        logger=logger
    )

    # 5. use trainer to optimize the model, given training and validation dataloader
    trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.test(dataloaders=test_dataloader, ckpt_path="last")
    predictions = trainer.predict(dataloaders=test_dataloader)


