import tensorflow as tf
import tensorflow.contrib.slim as slim
import resnet_v1


def res_v1_101_lstm(input_imgs, input_seqs, input_masks,
                    batch_size, embedding_size, vocab_size,
                    is_training, lstm_dropout_keep_prob):
    with tf.variable_scope('res_v1_101_lstm'):
        # Sequence embedding layer
        with tf.variable_scope("seq_embedding"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))

        # Image feature extraction layer
        with slim.arg_scope(resnet_v1.resnet_arg_scope(trainable=is_training)):
            # Set is_training = False to fix running mean/variance of batch normalization
            image_feature, _ = resnet_v1.resnet_v1_101(input_imgs, None, is_training=False, output_stride=32)

        # Image embedding layer
        image_feature = tf.squeeze(image_feature, axis=[1, 2])
        image_embedding = slim.fully_connected(image_feature, embedding_size, activation_fn=None,
                                               weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                                               biases_initializer=tf.zeros_initializer,
                                               weights_regularizer=slim.l2_regularizer(0.0005),
                                               scope = 'image_embedding')

        # LSTM layer
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=embedding_size, state_is_tuple=True)
        # Training process
        if is_training is True:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                      input_keep_prob=lstm_dropout_keep_prob,
                                                      output_keep_prob=lstm_dropout_keep_prob)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

            with tf.variable_scope("lstm", initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08)) as lstm_scope:
                # Feed the image embeddings to set the initial LSTM state.
                zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                _, initial_state = lstm_cell(image_embedding, zero_state)
                lstm_scope.reuse_variables()
                sequence_length = tf.reduce_sum(input_masks, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=seq_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)
                lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

            # Word logits layer
            output_logits = slim.fully_connected(lstm_outputs, vocab_size, activation_fn=None,
                                                weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                                                biases_initializer=tf.zeros_initializer,
                                                weights_regularizer=slim.l2_regularizer(0.0005),
                                                scope='logits'
                                                )

            variables = slim.get_variables('res_v1_101_lstm')
            res_variables = {}
            for variable in variables:
                if 'resnet_v1_101' in variable.name:
                    res_variables[variable.name[16:-2]] = variable

            return output_logits, res_variables
        # Inference process
        else:
            weights = tf.get_variable("logits/weights", [embedding_size, vocab_size])
            biases = tf.get_variable("logits/biases", [vocab_size])
            with tf.variable_scope("lstm", initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08)) as lstm_scope:
                # Feed the image embeddings to set the initial LSTM state.
                zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                _, initial_state = lstm_cell(image_embedding, zero_state)
                lstm_scope.reuse_variables()
                memory_state = initial_state
                output_words = [input_seqs[0]]
                # TODO: replace the end condition of the loop with meeting the end word
                for _ in range(30):
                    input_seqs = tf.nn.embedding_lookup(embedding_map, input_seqs)
                    output_seqs, memory_state = lstm_cell(input_seqs, memory_state)
                    output_logits = tf.matmul(output_seqs, weights) + biases
                    output_word = tf.argmax(output_logits, -1)
                    output_words.append(output_word[0])
                    input_seqs = output_word
                output_words = tf.stack(output_words)

            return output_words


