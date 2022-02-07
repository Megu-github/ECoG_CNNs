import datetime
import pytz

ROOT_DIRECTORY = '/home/megu/'
EXPT_DATE = str(datetime.datetime.now(pytz.timezone('Asia/Tokyo')).date())

class ParametersCommon:
    DATASET_CLASS = "image_folder"
    # Common settings
    classes = ('Anesthetized', 'EyesClosed')
    DEVICE = "cuda"

    # Training setting
    EPOCH = 200  # 100 or 5
    TEST_EPOCH = 100
    N_SPLITS = 5  # 5
    WEIGHT_DECAY = 0
    LEARNING_RATE = 0.0005  # 0.001 using adam?

    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'
    RESIZE = 224

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.5
    USE_DROPOUT = True
    USE_BATCH_NORM = False


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

class ParametersDebug2(ParametersCommon):
    """
    For debug after merging file
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 32
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = True
    USE_BATCH_NORM = True

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug3(ParametersCommon):
    """
    For debug after merging file
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 32
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug4(ParametersCommon):
    """
    For debug after merging file
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 32
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug5(ParametersCommon):
    """
    For debug after merging file
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug6(ParametersCommon):
    """
    Sarch optimizer, weight_decay, lr
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug7(ParametersCommon):
    """
    Search p_dropout1, p_dropout2
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = True
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug8(ParametersCommon):
    """
    For search use_batch_norm
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = True

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug9(ParametersCommon):
    """
    For search lr when adam and weight_decay=0
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class ParametersDebug10(ParametersCommon):
    """
    For search use_dropout and batch_norm
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters11(ParametersCommon):
    """
    For search lr when imagefolder, adam, weight_decay=0
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters12(ParametersCommon):
    """
    Search p_dropout1, p_dropout2
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/MK11_expt.2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = True
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters13(ParametersCommon):
    """
    main experiment about recoding date using Recording_Date1
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Recording_Date1'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Recording_Date1'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + DATASET_CLASS + '_' + OPTIMIZER_CLASS
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters14(ParametersCommon):
    """
    main experiment about recoding date using Recording_Date2
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Recording_Date2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Recording_Date2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Recoding_Date2"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Base_Line(ParametersCommon):
    """
    main experiment about base line using Base_Line
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Base_Line'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Base_Line'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Base_Line"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Recoding_Date_Chibi(ParametersCommon):
    """
    main experiment about recoding date using Recoding_Date_Chibi
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Recoding_Date_Chibi'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Recoding_Date_Chibi'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Recoding_Date_Chibi"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_Chibi(ParametersCommon):
    """
    main experiment about indiviual using Individual_Chibi
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_Chibi'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_Chibi'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Individual_Chibi"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_KTMD(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_KTMD
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KTMD'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KTMD'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_KTMD"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_George(ParametersCommon):
    """
    main experiment about indiviual using Individual_George
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_George'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_George'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Individual_George"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_Kin2(ParametersCommon):
    """
    main experiment about indiviual using Individual_Kin2
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_Kin2'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_Kin2'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Individual_Kin2"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_Su(ParametersCommon):
    """
    main experiment about indiviual using Individual_Su
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_Su'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Individual_Su'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Individual_Su"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_KT(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_KT
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KT'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KT'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_KT"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_MD(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_MD
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_MD'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_MD'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_MD"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_PF(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_PF
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_PF'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_PF'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.2
    USE_DROPOUT = False
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_PF"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER


class Parameters_BN_DropOut(ParametersCommon):
    """
    main experiment about BN and DropOut using Base Line
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Base_Line'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Base_Line'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.5
    USE_DROPOUT = True
    USE_BATCH_NORM = True

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Base_Line"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_KTMD_Dropout(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_KTMD with Dropout
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KTMD'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KTMD'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.5
    USE_DROPOUT = True
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_KTMD_Dropout"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_KT_Dropout(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_KT with Dropout
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KT'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_KT'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.5
    USE_DROPOUT = True
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_KT_Dropout"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_MD_Dropout(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_MD with Dropout
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_MD'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_MD'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.5
    USE_DROPOUT = True
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_MD_Dropout"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_PF_Dropout(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_PF with Dropout
    """
    # Directory and Dataset
    DATASET_CLASS = "image_folder"
    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_PF'
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/Anesthesia_PF'
    classes = ('Anesthetized', 'EyesClosed')

    # Training setting
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = 128
    OPTIMIZER_CLASS = 'adam'

    # Model parameters
    DEVICE = "cuda"
    P_DROPOUT1 = 0.2
    P_DROPOUT2 = 0.5
    USE_DROPOUT = True
    USE_BATCH_NORM = False

    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + "Anesthesia_PF_Dropout"
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Recording_Date_KTMD_Chibi_0621test(ParametersCommon):
    """
    main experiment about Recoding_Date using Recording_Date_KTMD_Chibi_0621test
    """
    # Directory and Dataset
    Dataset = "Recording_Date_KTMD_Chibi_0621test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Recording_Date_KTMD_George_20110112_test(ParametersCommon):
    """
    main experiment about Recoding_Date using Recording_Date_KTMD_George_20110112_test
    """
    # Directory and Dataset
    Dataset = "Recording_Date_KTMD_George_20110112_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Recording_Date_KTMD_George_20110113_test(ParametersCommon):
    """
    main experiment about Recoding_Date using Recording_Date_KTMD_George_20110113_test
    """
    # Directory and Dataset
    Dataset = "Recording_Date_KTMD_George_20110113_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER =  EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Recording_Date_KTMD_George_20110114_test(ParametersCommon):
    """
    main experiment about Recoding_Date using Recording_Date_KTMD_George_20110114_test
    """
    # Directory and Dataset
    Dataset = "Recording_Date_KTMD_George_20110114_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER =  EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_Chibi_2(ParametersCommon):
    """
    main experiment about Individual using Individual_Chibi_2
    """
    # Directory and Dataset
    Dataset = "Individual_Chibi_2"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_George_2(ParametersCommon):
    """
    main experiment about Individual using Individual_George_2
    """
    # Directory and Dataset
    Dataset = "Individual_George_2"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_Kin2_2(ParametersCommon):
    """
    main experiment about Individual using Individual_Kin2_2
    """
    # Directory and Dataset
    Dataset = "Individual_Kin2_2"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Individual_Su_2(ParametersCommon):
    """
    main experiment about Individual using Individual_Su_2
    """
    # Directory and Dataset
    Dataset = "Individual_Su_2"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_KTMD_test(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_KTMD_test
    """
    # Directory and Dataset
    Dataset = "Anesthesia_KTMD_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_PF_test(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_PF_test
    """
    # Directory and Dataset
    Dataset = "Anesthesia_PF_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_MD_test(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_MD_test
    """
    # Directory and Dataset
    Dataset = "Anesthesia_MD_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_Anesthesia_KT_test(ParametersCommon):
    """
    main experiment about Anesthesia using Anesthesia_KT_test
    """
    # Directory and Dataset
    Dataset = "Anesthesia_KT_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER


class Parameters_Recording_Date_KTMD_Chibi_20110622_test(ParametersCommon):
    """
    main experiment about Anesthesia using Recording_Date_KTMD_Chibi_20110622_test
    """
    # Directory and Dataset
    Dataset = "Recording_Date_KTMD_Chibi_20110622_test"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_frequency_KTMD(ParametersCommon):
    """
    main experiment about frequency using frequency_KTMD
    """
    # Directory and Dataset
    Dataset = "frequency_KTMD"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_frequency_KT(ParametersCommon):
    """
    main experiment about frequency using frequency_KT
    """
    # Directory and Dataset
    Dataset = "frequency_KT"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER =  EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_frequency_MD(ParametersCommon):
    """
    main experiment about frequency using frequency_MD
    """
    # Directory and Dataset
    Dataset = "frequency_MD"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = EXPT_DATE + '_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER

class Parameters_frequency_PF(ParametersCommon):
    """
    main experiment about frequency using frequency_PF
    """
    # Directory and Dataset
    Dataset = "frequency_PF"

    TRAIN_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    TEST_DATASET_PATH = ROOT_DIRECTORY + 'CNN_Dataset/' + Dataset
    classes = ('Anesthetized', 'EyesClosed')


    # Save
    EXPT_NUMBER = '2022-02-05_' + Dataset
    RESULT_DIR_PATH = ROOT_DIRECTORY + 'ECoG_CNNs/Result/' + EXPT_NUMBER
