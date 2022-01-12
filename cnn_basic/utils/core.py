import numpy as np
import time


def convert_cat2one_hot(y: np.array)->np.array:
    one_hot_mat=np.zeros((y.size,y.max()+1))
    one_hot_mat[np.arange(y.size),y]=1
    return one_hot_mat


def convert_prob2cat(probs: np.array)->np.array:
    return np.argmax(probs,axis=1)


def convert_prob2one_hot(probs):
    class_idx=convert_prob2cat(probs)
    one_hm=np.zeros_like(probs)
    one_hm[np.arange(probs.shape[0]),class_idx]=1
    return one_hm


def generate_batches(x: np.array, y: np.array, batch_size: int):
    """
    :param x - features array with (n, ...) shape
    :param y - one hot ground truth array with (n, k) shape
    :batch_size - number of elements in single batch
    ----------------------------------------------------------------------------
    n - number of examples in data set
    k - number of classes
    """
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(
                i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(
                i, min(i + batch_size, y.shape[0])), axis=0)
        )


def format_time(start_time: time.time, end_time: time.time) -> str:
    """
    :param start_time - beginning of time period
    :param end_time - ending of time period
    :output - string in HH:MM:SS.ss format
    ----------------------------------------------------------------------------
    HH - hours
    MM - minutes
    SS - seconds
    ss - hundredths of a second
    """
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)