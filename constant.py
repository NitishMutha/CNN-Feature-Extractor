## params

# Reward Settings
REWARD_SALIENCY = True
REWARD_SEGMENTATION = False

# videographerA3C
EPISODE_BUFFER = 20

# Saliency
SAL_THRESHOLD = 0.9

# video360Env
ENV_HEIGHT = 400
ENV_WIDTH = 800


EYE_ROOT = "brain/"
HTML = "html/"

AROUND = "index.html"
NAMESPACE = "/test"

EQUIRECTANGULAR = 'equirectangular'
CUBEMAP = 'cubemap'

# modes
TRAIN = 'train'
TEST = 'test'

# models
RESNET152 = 'resnet152'

# ckpts
RESNET_152_CKPT = 'resnet/resnet_v2_152.ckpt'
VGG_PRETRAIN_MODEL = 'vgg16/vgg16.npy'

# model scopes
CINE = 'cinematography'
VIDRNN = 'videographyRNN_'

GLOBAL = 'global'

# videos categories
CAT1 = 'dancing'
CAT2 = 'fight'

# placeholders
FRAME_FEATURE_INPUTS = 'frame_feature_inputs'

# Dirs
PRETRAINED_ROOT = '../pretrainedWeightsRepo/'
DATA_DIR = '../segmentation_data/images/'
VIDEO_DATA_FOLDER = '../dataset/source/'
FRAMES_DATA_FOLDER = '../dataset/extracted/'

MODEL_PATH = '../pretrainedWeightsRepo/trained_model/'
