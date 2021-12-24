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
    ROOT_DIRECTORY = '/new_nas/common/kobayashi_211222/'
    parameter_common = parameters.ParametersDebug1
    parameter_list = itertools.product(
        ['my_dataset', 'image_folder'],
        [['adam', 0.001], ['sgd', 0.0001]],
        [True, False],
    )

    for idx, parameter_change in enumerate(parameter_list):
        print("####################")
        print("####################")
        print("condition: ", str(idx), parameter_change)

        parameter = parameter_common
        parameter.DATASET_CLASS = parameter_change[0]
        parameter.OPTIMIZER_CLASS = parameter_change[1][0]
        parameter.LEARNING_RATE = parameter_change[1][1]
        parameter.USE_DROPOUT = parameter_change[2]
        parameter.RESULT_DIR_PATH = ROOT_DIRECTORY + \
            'ECoG_CNNs/Result/211224/' + '_'.join([
                str(param) for param in parameter_change])

        run_one_condition(parameter=parameter)
    print("all done")


if __name__ == "__main__":
    # learning(parameter=parameters.Parameters1)
    # test_smoothgrad(parameter=parameters.Parameters1)
    run_one_condition(parameter=parameters.Parameters1)
    # run_grid_search()
    # run_grid_search_single_comparing()
