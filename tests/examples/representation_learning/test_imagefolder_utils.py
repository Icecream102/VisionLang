from examples.representation_learning.imagefolder_utils import stratified_split_indices


def test_stratified_split_has_each_class_in_both_sets():
    train, val = stratified_split_indices([0] * 10 + [1] * 10, 0.2, seed=42)

    assert len(train) == 16
    assert len(val) == 4
    assert {0, 1}.issubset({0 if index < 10 else 1 for index in train})
    assert {0, 1}.issubset({0 if index < 10 else 1 for index in val})
