def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score based on true and predicted labels.

    Args:
        y_true (list): True labels (ground truth).
        y_pred (list): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    # Your code here
    tp = fp = fn = 0
    for actual, predicted in zip(y_true,y_pred):
        if actual == 1  and predicted == 1:
            tp+=1
        elif actual == 1 and predicted == 0:
            fn+=1
        elif actual == 0 and predicted == 1:
            fp+=1
        
    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0

    f1 = (2*precision*recall)/(precision+recall) if (precision*recall) else 0.0
    return round(f1,3)