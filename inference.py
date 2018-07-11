import tensorflow as tf
import model
import utils
import cv2

# Define hyperparameters
embedding_size = 512
vocab_size = 9955
img_dir = '2.jpg'
vocab_dir = 'vocabulary.csv'
res_v1_101_lstm_parameters_dir = 'res_v1_101_lstm.ckpt'
# RGB mean value for images
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


# Preprocess the data, move to zero mean
image = tf.placeholder(tf.uint8, [None, None, None, 3])
image = tf.cast(image, tf.float32)
image = image - [_R_MEAN, _G_MEAN, _B_MEAN]
input_seqs = tf.constant([1], dtype=tf.int64)
vocab = utils.read_csv_file(vocab_dir)


# Build model and predict outputs sequences
output_words = model.res_v1_101_lstm(image, input_seqs, None, 1, embedding_size, vocab_size, False, 0.5)


# Initialization
local_vars_init_op = tf.local_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(local_vars_init_op)
    saver.restore(sess, res_v1_101_lstm_parameters_dir)
    img_ori = cv2.imread(img_dir)
    img = cv2.resize(img_ori, (512, 512))
    sentence = sess.run(output_words, feed_dict={image: [img]})
    sentence = [vocab[word_id] for word_id in sentence]
    caption = ""
    for word in sentence:
        caption += word + " "
        if word == '</S>': break
    caption = caption[3:-5]
    print caption
    cv2.putText(img_ori, caption, (0,25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255))
    cv2.imshow("image", img_ori)
    cv2.waitKey(0)
