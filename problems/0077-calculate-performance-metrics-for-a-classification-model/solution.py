
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    tp = fp = fn = tn = 0

    for y_true, y_pred in zip(actual, predicted):
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        else:
            tn += 1

    confusion_matrix = [[tp, fn], [fp, tn]]

    total = tp + fn + fp + tn

    accuracy = (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    negativePredictive = tn / (tn + fn) if (tn + fn) else 0.0

    return (
        confusion_matrix,
        round(accuracy, 3),
        round(f1, 3),
        round(specificity, 3),
        round(negativePredictive, 3)
    )
