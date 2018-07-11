import tensorflow as tf
import os, sys

fixed_len = tf.constant(30, dtype=tf.int32)

def read_data_from_tf_records(data_dir, is_training, batch_size):
    # Build the filename queue
    if is_training is True:
        filename_queue = tf.train.string_input_producer(data_dir, shuffle=True)
    else:
        filename_queue = tf.train.string_input_producer(data_dir, num_epochs=1)

    reader = tf.TFRecordReader()
    # Read the sequence example from tfrecords file
    _, serialized_example = reader.read(filename_queue)

    # Parse the sequence example back to raw data
    context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        context_features={
            'caption_length': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            'caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        })

    caption_len = context['caption_length']
    height = context['height']
    width = context['width']
    encoded_image = context['image_raw']
    caption = sequence['caption_ids']

    # Decode and process raw data
    caption_len = tf.cast(caption_len, tf.int32)
    image = tf.decode_raw(encoded_image, tf.uint8)
    height = tf.cast(height, tf.int32)
    width = tf.cast(width, tf.int32)
    caption = tf.reshape(caption, tf.stack([caption_len]))

    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)

    if is_training is True:
        # Generate input, target and mask
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                       target_height=512,
                                                       target_width=512)
        input_words, target_words, input_mask = generate_input_with_target(caption, caption_len, fixed_len)

        images, input_seqs, target_seqs, mask = tf.train.shuffle_batch(
            [image, input_words, target_words, input_mask],
            batch_size=batch_size,
            capacity=3000,
            num_threads=3,
            min_after_dequeue=1000,
            allow_smaller_final_batch=True)

        return images, input_seqs, target_seqs, mask
    else:
        caption = tf.reshape(caption, tf.stack([caption_len]))
        return image, caption


def generate_input_with_target(caption, caption_len, fixed_len):
    valid_length = tf.cond(tf.less(caption_len - 2, fixed_len), lambda: caption_len, lambda: fixed_len + 1)
    input_words = caption[0:valid_length-1]
    target_words = caption[1:valid_length]
    input_words = tf.reshape(input_words, tf.stack([valid_length - 1]))
    input_words = tf.pad(input_words, paddings=tf.stack([tf.stack([0, fixed_len - valid_length + 1])]))
    input_words = tf.reshape(input_words, tf.stack([fixed_len]))
    target_words = tf.reshape(target_words, tf.stack([valid_length - 1]))
    target_words = tf.pad(target_words, paddings=tf.stack([tf.stack([0, fixed_len - valid_length + 1])]))
    target_words = tf.reshape(target_words, tf.stack([fixed_len]))
    input_mask = tf.ones(tf.stack([valid_length - 1]), dtype=tf.int64)
    input_mask = tf.pad(input_mask, paddings=tf.stack([tf.stack([0, fixed_len - valid_length + 1])]))
    input_mask = tf.reshape(input_mask, tf.stack([fixed_len]))

    return input_words, target_words, input_mask