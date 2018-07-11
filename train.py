import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
import read_data

# Define hyperparameters
batch_size = 4
embedding_size = 512
vocab_size = 9955
learning_rate = 0.00001
training_rounds = 300000
data_dir = 'train2014.tfrecords'
res_v1_101_parameters_dir = 'resnet_v1_101.ckpt'
res_v1_101_lstm_parameters_dir = 'res_v1_101_lstm.ckpt'
# RGB mean value for images
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


# Read batches of data
images, input_seqs, target_seqs, mask = read_data.read_data_from_tf_records([data_dir], True, batch_size)
# Preprocess the data, move to zero mean
images = tf.cast(images, tf.float32)
images = images - [_R_MEAN, _G_MEAN, _B_MEAN]


# Build model and predict outputs sequences
logits_seqs, res_variables = model.res_v1_101_lstm(images, input_seqs, mask, batch_size, embedding_size, vocab_size, True, 0.5)


# Calculate loss based on true target sequences
target_seqs = tf.reshape(target_seqs, [-1])
mask_seqs = tf.cast(tf.reshape(mask, [-1]), tf.float32)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_seqs, logits=logits_seqs)
loss = tf.div(tf.reduce_sum(tf.multiply(loss, mask_seqs)), tf.reduce_sum(mask_seqs))
preds = tf.arg_max(logits_seqs, dimension=-1)
accuracy = tf.div(tf.reduce_sum(tf.multiply(mask_seqs, tf.cast(tf.equal(preds, target_seqs), tf.float32))), tf.reduce_sum(mask_seqs))


# Adam Optimizer used for minimizing loss function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)


# Initialization
pretrained_init_op = slim.assign_from_checkpoint_fn(res_v1_101_parameters_dir, res_variables)
global_vars_init_op = tf.global_variables_initializer()
local_vars_init_op = tf.local_variables_initializer()
combined_op = tf.group(local_vars_init_op, global_vars_init_op)
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(combined_op)
    pretrained_init_op(sess)
    # If restart training after break
    saver.restore(sess, res_v1_101_lstm_parameters_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(training_rounds):
        _, error, accu  = sess.run([optimizer, loss, accuracy])
        print("Round %d, Loss = %f, Accuracy = %f" % (i, error, accu))
        if i % 299 == 0:
            saver.save(sess, res_v1_101_lstm_parameters_dir)
    coord.request_stop()
    coord.join(threads)

saver.save(sess, res_v1_101_lstm_parameters_dir)