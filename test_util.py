from itertools import product


def test_hparamenter_count():
    gamma_ranges = [0.001, 0.01, 0.1, 1]
    C_ranges = [0.1, 1, 2, 5, 10]
    hparam_combos = product(gamma_ranges, C_ranges)
    print(hparam_combos)
    assert 20 == len(gamma_ranges)*len(C_ranges)



def test_in_gamma_ranges():
    gamma_ranges = [0.001, 0.01, 0.1, 1]
    ele = 0.1
    assert ele in gamma_ranges