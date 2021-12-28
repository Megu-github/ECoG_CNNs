import os
import itertools

from test_smoothgrad import test_smoothgrad
from learning import learning
import parameters


def run_one_condition(parameter):
    # make result dir
    os.makedirs(parameter.RESULT_DIR_PATH, exist_ok=True)
    file_path = parameter.RESULT_DIR_PATH + "/" + \
        parameter.EXPT_NUMBER + '.log'

    with open(file_path, 'a') as f:
        print("Parameter:\n", file=f)
        for key, val in vars(parameter).items():
            print(key, ':\t', val, file=f)
    print("start learning")
    learning(parameter=parameter)
    print("start test and visualizing")
    test_smoothgrad(parameter=parameter)


def run_grid_search():
    parameter_list = [
        parameters.ParametersDebug1,
        parameters.Parameters1,
        parameters.Parameters2,
    ]
    for idx, parameter in enumerate(parameter_list):
        print("####################")
        print("####################")
        print("condition: ", str(idx))
        run_one_condition(parameter=parameter)
    print("all done")


def run_grid_search_single_comparing():
    parameter_common = parameters.ParametersDebug10
    use_parameter_dict_key = [
        #'optimizer',
        #'weight_decay',
        #'lr',
        'use_dropout',
        'use_batch_norm',
        #'p_dropout1',
        #'p_dropout2',
        ]
    parameter_dict = {
        'dataset': ['my_dataset', 'image_folder'],
        'optimizer': ['adam',],
        'weight_decay': [0,],
        'lr': [ 0.0005, 0.0001, 0.00005,],
        'use_dropout': [False, True],
        'use_batch_norm': [False, True],
        'p_dropout1': [0,1, 0.2, 0.3],
        'p_dropout2': [0.3, 0.4, 0.5],
    }
    parameter_dict_value = []
    for idx in range(len(use_parameter_dict_key)):
        parameter_dict_value.append(
            parameter_dict[
                use_parameter_dict_key[idx]
                ]
            )

    parameter_list = itertools.product(
        *parameter_dict_value
    )

    for idx, parameter_change in enumerate(parameter_list):
        print("####################")
        print("####################")
        print("condition: ", str(idx), parameter_change)
        parameter_change_dict = dict(
            zip(use_parameter_dict_key, parameter_change)
            )

        parameter = parameter_common
        # parameter.DATASET_CLASS = parameter_change_dict['dataset']
        #parameter.OPTIMIZER_CLASS = parameter_change_dict['optimizer']
        #parameter.WEIGHT_DECAY = parameter_change_dict['weight_decay']
        #parameter.LEARNING_RATE = parameter_change_dict['lr']

        parameter.USE_DROPOUT = parameter_change_dict['use_dropout']
        parameter.USE_BATCH_NORM = parameter_change_dict['use_batch_norm']
        #parameter.P_DROPOUT1 = parameter_change_dict['p_dropout1']
        #parameter.P_DROPOUT2 = parameter_change_dict['p_dropout2']
        parameter.RESULT_DIR_PATH = parameters.ROOT_DIRECTORY + \
            'ECoG_CNNs/Result/' + parameters.EXPT_DATE + '/' + '_'.join([
                str(param) for param in parameter_change])

        run_one_condition(parameter=parameter)
    print("all done")


if __name__ == "__main__":
    # learning(parameter=parameters.Parameters1)
    # test_smoothgrad(parameter=parameters.Parameters1)
    # run_one_condition(parameter=parameters.ParametersDebug3)
    # run_grid_search()
    run_grid_search_single_comparing()
