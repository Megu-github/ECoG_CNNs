ROOT_DIRECTORY = '/new_nas/common/kobayashi_211222/'


class ParametersCommon:
    # Common settings
    classes = ('Anesthetized', 'EyesClosed')
    DEVICE = "cuda"

    # Training setting
    EPOCH = 10  # 100 or 5
    TEST_EPOCH = EPOCH
    N_SPLITS = 2  # 5
    WEIGHT_DECAY = 0.005
    LEARNING_RATE = 0.001  # 0.001 using adam?

    # Model parameters
    RESIZE = 224


class Parameters1(ParametersCommon):
    """
    Using image_folder
    """

    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    EXPT_NUMBER = 'MK11_MK3_' + DATASET_CLASS
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK3_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'

    # Training setting
    TRAIN_BATCH_SIZE = 1000
    TEST_BATCH_SIZE = 2
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    P_DROPOUT1 = 0.5
    P_DROPOUT2 = 0.5
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = 'test_param1_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER


class Parameters2(ParametersCommon):
    """
    Using my_dataset
    """

    # Directory and Dataset
    DATASET_CLASS = "my_dataset"
    EXPT_NUMBER = 'MK11_MK3_' + DATASET_CLASS
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK3_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'

    # Training setting
    TRAIN_BATCH_SIZE = 1000
    TEST_BATCH_SIZE = 2
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    P_DROPOUT1 = 0.5
    P_DROPOUT2 = 0.5
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = 'test_param2_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER


class ParametersDebug1(ParametersCommon):
    """
    For debug using small datasize
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Debug_Dataset'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Debug_Dataset'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 2
    TEST_BATCH_SIZE = 2
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = True
    USE_BATCH_NORM = True

    # Save
    EXPT_NUMBER = 'test_debug_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER
