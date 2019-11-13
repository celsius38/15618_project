import time
import numpy as np

eps = 1
minPts = 10

def timeit(fun):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fun(*args, **kwargs)
        print("taking {time.time() - start}s")
        return res
    return wrapper

def dbscan_ref(in_file, eps, min_pts):
    from sklearn.cluster import DBSCAN
    points = np.loadtxt(in_file)
    clustering = DBSCAN(eps=eps, min_samples=min_pts).fit(points)
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return num_clusters, labels

def check_label(gold_num_clusters, gold_labels, num_clusters, labels):
    """
    check if the label output match with that of gold
    notice that same cluster might have different label in different run
    """
    if len(gold_labels) != len(labels):
        print("length {len(labels)}, expected {len(gold_labels)}")
        return False
    if gold_num_clusters != num_clusters:
        print("{num_clusters} clusters, expected {gold_num_clusters}")
        return False
    mapping = {}  # gold_label -> label
    for gold, label in zip(gold_labels, labels):
        if gold == -1:
            if label != -1:
                return False
        else:
            if gold in mapping:
                if label != mapping[gold]:
                    return False
            else:
                mapping[gold] = label
    return True

