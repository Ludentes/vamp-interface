from arc_distill.dataset import is_held_out


def test_is_held_out_f_prefix_held():
    assert is_held_out("f1234567" + "0" * 56) is True


def test_is_held_out_other_prefix_train():
    assert is_held_out("a1234567" + "0" * 56) is False
    assert is_held_out("01234567" + "0" * 56) is False


def test_is_held_out_uppercase_ok():
    assert is_held_out("F1234567" + "0" * 56) is True
