class Parameters1():
    WEIGHT_DECAY = 0.005
    LEARNING_RATE = 0.0001
    RESIZE = [224, 224]
    DEVICE = "cuda" # サーバー上なら"cuda"
    EXPT_NUMBER = 'Debug_test'
    RESULT_DIR_PATH = '/home/megu/ECoG_CNNs/Result/' + EXPT_NUMBER


    TEST_DATASET_PATH = '/home/megu/CNN_Dataset/MK3_expt.1' # セーバーにDATASETをコピーして、そのpathを書く
    TEST_BATCH_SIZE = 5

    TRAIN_DATASET_PATH = '/home/megu/CNN_Dataset/MK3_expt.1' # セーバーにDATASETをコピーして、そのpathを書く
    TRAIN_BATCH_SIZE = 2
    EPOCH = 10

    classes = ('Anesthetized', 'EyesClosed')

class Parameters2():
    WEIGHT_DECAY = 0.005
    LEARNING_RATE = 0.0001
    RESIZE = [224, 224]
    DEVICE = "cuda" # サーバー上なら"cuda"
    EXPT_NUMBER = 'Debug_test'
    RESULT_DIR_PATH = '/home/megu/ECoG_CNNs/Result/' + EXPT_NUMBER


    TEST_DATASET_PATH = '/home/megu/CNN_Dataset/Debug_Dataset' # セーバーにDATASETをコピーして、そのpathを書く
    TEST_BATCH_SIZE = 2

    TRAIN_DATASET_PATH = '/home/megu/CNN_Dataset/Debug_Dataset' # セーバーにDATASETをコピーして、そのpathを書く
    TRAIN_BATCH_SIZE = 2
    EPOCH = 3

    classes = ('Anesthetized', 'EyesClosed')
