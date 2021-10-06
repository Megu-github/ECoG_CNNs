
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
RESIZE = [224, 224]
DEVICE = "cuda" # サーバー上なら"cuda"

TEST_DATASET_PATH = '/home/megu/CNN_Dataset/MK1_expt.3' # セーバーにDATASETをコピーして、そのpathを書く
TEST_EXPT_NUMBER = 'move_test'
TEST_BATCH_SIZE = 4

TRAIN_DATASET_PATH = '/home/megu/CNN_Dataset/MK1_expt.3' # セーバーにDATASETをコピーして、そのpathを書く
TRAIN_EXPT_NUMBER = 'move_test'
TRAIN_BATCH_SIZE = 16
EPOCH = 20