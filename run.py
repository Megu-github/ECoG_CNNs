from learning import learning
from parameters import Parameters1, Parameters2


def grid_search():
    parameter_list = [Parameters1, Parameters2]
    for parameter in parameter_list:
        learning(parameter=parameter)

    return
