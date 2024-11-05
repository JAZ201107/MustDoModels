from yacs.config import CfgNode as CN

__C = CN()

# Model arguments configuration
__C.MODEL = CN()
__C.MODEL.DIM = 4096
__C.MODEL.N_LAYERS = 32
__C.MODEL.N_HEADS = 32
__C.MODEL.N_KV_HEADS = None
__C.MODEL.VOCAB_SIZE = -1  # Later set in the build method
__C.MODEL.MULTIPLE_OF = 256
__C.MODEL.FFN_DIM_MULTIPLIER = None
__C.MODEL.NORM_EPS = 1e-5
__C.MODEL.MAX_BATCH_SIZE = 32
__C.MODEL.MAX_SEQ_LEN = 2048
