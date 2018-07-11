import tensorflow as tf
import csv


def softmax_cross_entropy_loss(logits, labels, masks):
    """Calculcate softmax cross entropy loss between logits and labels
       based on the masking"""
    e = 10e-6
    softmax_preds = tf.nn.softmax(logits, dim=-1)
    loss = tf.reduce_mean(
               tf.multiply(
                   tf.reduce_sum(
                       -tf.multiply(labels, tf.log(softmax_preds + e)) -
                        tf.multiply(tf.subtract(1., labels),
                                    tf.log(tf.subtract(1., softmax_preds) + e)),
                        axis=-1),
                   masks),
               axis=[0, 1])
    return loss


def read_csv_file(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        mydict = {}
        for row in reader:
            k, v = row
            mydict[int(k)] = v
        return mydict