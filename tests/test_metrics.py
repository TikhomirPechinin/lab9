import pytest
from ai_utils.metrics import accuracy, precision, recall, f1_score

def test_accuracy():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 1]
    assert accuracy(y_true, y_pred) == 0.75

def test_accuracy_perfect():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    assert accuracy(y_true, y_pred) == 1.0

def test_accuracy_empty():
    with pytest.raises(ValueError):
        accuracy([], [])

def test_accuracy_different_length():
    with pytest.raises(ValueError):
        accuracy([1, 2], [1])

def test_precision():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 1, 0]
    assert precision(y_true, y_pred) == 2/3

def test_recall():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 1, 0]
    assert recall(y_true, y_pred) == 2/3

def test_f1_score():
    assert f1_score([1, 0, 1, 0], [1, 0, 1, 1]) == 0.8