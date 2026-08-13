def recall_at_5(results, truth): return float(any(x["defect_id"]==truth for x in results[:5]))
def average_precision_at_5(results, truth):
    hits=0; total=0.0
    for rank,x in enumerate(results[:5],1):
        if x["defect_id"]==truth: hits+=1; total+=hits/rank
    return total/min(5, max(1,hits)) if hits else 0.0
def ndcg_at_5(results, truth):
    import math
    dcg=sum((1/math.log2(i+2)) for i,x in enumerate(results[:5]) if x["defect_id"]==truth)
    ideal=sum(1/math.log2(i+2) for i in range(min(5, sum(x["defect_id"]==truth for x in results[:5]))))
    return dcg/ideal if ideal else 0.0
