import math

def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    
    max_scores = max(scores)
    exp_scores = [math.exp(x-max_scores) for x in scores]
    total = sum(exp_scores)

    return [x/total for x in exp_scores]



    pass