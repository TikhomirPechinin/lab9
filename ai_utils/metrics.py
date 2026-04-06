"""
Модуль с метриками качества для задач классификации.
Содержит функции: accuracy, precision, recall, f1_score.
"""

def accuracy(y_true: list, y_pred: list) -> float:
    """
    Вычисляет точность (accuracy) классификации.

    Parameters:
        y_true (list): Истинные метки
        y_pred (list): Предсказанные метки

    Returns:
        float: Значение accuracy (от 0 до 1)

    Raises:
        ValueError: Если списки разной длины или пустые

    Example:
        >>> accuracy([1, 0, 1, 0], [1, 0, 1, 1])
        0.75
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Списки должны быть одинаковой длины")
    if len(y_true) == 0:
        raise ValueError("Список не должен быть пустым")
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def precision(y_true: list, y_pred: list, positive_class=1) -> float:
    """
    Вычисляет точность (precision) для бинарной классификации.

    Parameters:
        y_true (list): Истинные метки
        y_pred (list): Предсказанные метки
        positive_class: Значение положительного класса (по умолчанию 1)

    Returns:
        float: Значение precision (от 0 до 1)
    """
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == positive_class and pred == positive_class)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true != positive_class and pred == positive_class)
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: list, y_pred: list, positive_class=1) -> float:
    """
    Вычисляет полноту (recall) для бинарной классификации.

    Parameters:
        y_true (list): Истинные метки
        y_pred (list): Предсказанные метки
        positive_class: Значение положительного класса (по умолчанию 1)

    Returns:
        float: Значение recall (от 0 до 1)
    """
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == positive_class and pred == positive_class)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == positive_class and pred != positive_class)
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: list, y_pred: list, positive_class=1) -> float:
    """
    Вычисляет F1-меру (гармоническое среднее precision и recall).

    Parameters:
        y_true (list): Истинные метки
        y_pred (list): Предсказанные метки
        positive_class: Значение положительного класса (по умолчанию 1)

    Returns:
        float: Значение F1-меры (от 0 до 1)
    """
    p = precision(y_true, y_pred, positive_class)
    r = recall(y_true, y_pred, positive_class)
    
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)