from yacs.config import CfgNode as CF


def get_config():
    __C = CF()

    # Training
    __C.TRAINING = CF()
    __C.TRAINING.BATCH_SIZE = 8
    __C.TRAINING.NUM_EPOCHS = 20
    __C.TRAINING.LEARNING_RATE = 1e-4
    __C.TRAINING.PRELOAD = "latest"

    # DATA
    __C.DATA = CF()
    __C.DATA.LANG_SRC = "en"
    __C.DATA.LANG_TGT = "it"
    __C.DATA.TOKENIZER_FILE = "tokenizer_{0}.json"

    # MODEL
    __C.MODEL = CF()
    __C.MODEL.D_MODEL = 512

    # EXPERIMENT
    __C.EXPERIMENT = CF()
    __C.EXPERIMENT.NAME = "runs/tmodel"
    __C.EXPERIMENT.MODEL_FOLDER = "weights"
    __C.EXPERIMENT.MODEL_BASENAME = "tmodel_"

    return __C
