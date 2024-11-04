from yacs.config import CfgNode as CF


__C = CF()


# MODEL
__C.MODEL = CF()
__C.MODEL.IN_CHANNELS = 3
__C.MODEL.OUT_CHANNELS = 1
__C.MODEL.FEATURES = [64, 128, 256, 512]

# DATA
__C.DATA = CF()
__C.DATA.TRAIN_IMAGE_DIR = "data/images"
__C.DATA.TRAIN_MASK_DIR = "data/masks"
__C.DATA.VAL_IMAGE_DIR = "data/val_images"
__C.DATA.VAL_MASK_DIR = "data/val_masks"


# TRAIN
__C.TRAIN = CF()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.LR = 1e-4
__C.TRAIN.NUM_EPOCHS = 5
__C.TRAIN.NUM_WORKERS = 2
__C.TRAIN.PIN_MEMORY = True


def get_config():
    return __C.clone()
