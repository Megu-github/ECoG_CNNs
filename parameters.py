
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
RESIZE = [224, 224]
DEVICE = "cuda" # サーバー上なら"cuda"
EXPT_NUMBER = 'MK18_expt.0_5'
result_dir_path = '/home/megu/ECoG_CNNs/Result/' + EXPT_NUMBER

TEST_DATASET_PATH = '/home/megu/CNN_Dataset/MK11_expt.1' # セーバーにDATASETをコピーして、そのpathを書く
TEST_BATCH_SIZE = 64

TRAIN_DATASET_PATH = '/home/megu/CNN_Dataset/MK11_expt.1' # セーバーにDATASETをコピーして、そのpathを書く
TRAIN_BATCH_SIZE = 64
EPOCH = 10
