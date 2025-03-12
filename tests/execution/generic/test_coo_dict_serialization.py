import execution.generic.coo_dict_serialization as dnl
import os


def test_coo_dict_serialization():
    coo_dict = {
        1: {2: 1, 3: 2},
        2: {1: 3, 3: 4},
    }
    filename = "coo_dict_serialization_test.npy"
    dnl.save_ddnl(coo_dict, filename)
    loaded_coo_dict = dnl.load_ddnl(filename)
    print(coo_dict)
    print(loaded_coo_dict)
    assert coo_dict == loaded_coo_dict
    os.remove(filename)
