import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import numpy as np


class CarvanDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir, self.images[idx].replace(".jpg", "_mask.gif")
        )

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def get_dataloader(config, train_transform, val_transform):
    train_ds = CarvanDataset(
        config.DATA.TRAIN_IMAGE_DIR,
        config.DATA.TRAIN_MASK_DIR,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEMORY,
    )

    val_ds = CarvanDataset(
        config.DATA.VAL_IMAGE_DIR,
        config.DATA.VAL_MASK_DIR,
        transform=val_transform,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=config.TRAIN.PIN_MEMORY,
    )

    return {"train_dataloader": train_dataloader, "val_dataloader": val_dataloader}
