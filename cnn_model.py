# Reference to http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

"""
Trains a CNN for text classification.

"""

import tensorflow as tf


random_seed = 1337
embedding_dim = 200
filter_sizes = '2,3,4,5,6,7'  # WARNING: must be same as max(legal_template_sizes) backup 2,3,4,5,6,7
num_filters = 128  # 128
l2_reg_lambda = 0.2
dropout_keep_prob = 0.5
batch_size = 32
num_epochs = 50
word2vec_path = None  # 'data/word2vec/labeledTrainData_sentences_word2vec.txt'
evaluate_every = 5000
max_sequence_length = 3000

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", embedding_dim, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", filter_sizes, "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", num_filters, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", dropout_keep_prob, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", l2_reg_lambda, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", num_epochs, "Number of training epochs (default: 200)")
tf.flags.DEFINE_string("word2vec_path", word2vec_path, "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("evaluate_every", evaluate_every, "Evaluate model on dev set after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


class TextCNNModel(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_each_filter, l2_reg_lambda=l2_reg_lambda):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),  # initialize randomly
                                 trainable=True, name="W")  # DO NOT train embeddings
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  # Add a dimension at last

        # Add dropout for embedding features
        with tf.name_scope("dropout_embedding"):
            self.h_drop_embedded = tf.nn.dropout(self.embedded_chars_expanded, self.dropout_keep_prob)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_each_filter]
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                filter_W = tf.Variable(tf.random_uniform(filter_shape, -0.05, 0.05), name="W")
                b = tf.Variable(tf.constant(0., shape=[num_each_filter]), name="b")
                conv = tf.nn.conv2d(self.h_drop_embedded,
                                    filter_W,
                                    strides=[1, 1, filter_size, 1],  # [1, filter_size, filter_size, 1]
                                    padding="VALID",
                                    name="conv_%s" % filter_size)
                conv_out = tf.nn.bias_add(conv, b)

                print(conv_out)

                # Apply nonlinearity
                # h = tf.nn.relu(conv_out, name="relu")
                # Apply leaky ReLU
                h = tf.maximum(0.2 * conv_out, conv_out, name="leaky_relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    # ksize=[1, sequence_length / filter_size, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_each_filter * len(filter_sizes)
        print(num_filters_total)
        print(pooled_outputs)
        self.h_pool = tf.concat(pooled_outputs, 3)
        print(self.h_pool)
        print('The shape of self.h_pool is {shape}'.format(shape=tf.shape(self.h_pool)))
        print('The first size of self.h_pool is {dimension} (dimension=tf.shape(self.h_pool)[0])'.format(dimension=tf.shape(self.h_pool)[0]))
        print('The first size of self.h_pool is {dimension} (self.h_pool.get_shape().as_list())'.format(dimension=self.h_pool.get_shape().as_list()))
        # self.h_pool_flat = tf.reshape(self.h_pool, [tf.shape(self.h_pool)[0], num_filters_total])
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print(self.h_pool_flat)

        # Add dropout for conv features
        with tf.name_scope("dropout_conv"):
            self.h_drop_conv = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # MLP layer 1
        with tf.name_scope("mlp_1"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, int(num_filters_total / 2)], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0., shape=[int(num_filters_total / 2)]), name="b")
            # mlp_1_h = tf.nn.xw_plus_b(self.h_drop_conv, W, b, name="mlp_1_h")
            mlp_1_h = tf.nn.xw_plus_b(self.h_drop_conv, W, b, name="mlp_1_h")
            # Apply nonlinearity
            # self.mlp_1_output = tf.nn.relu(mlp_1_h)
            self.mlp_1_output = tf.maximum(0.2 * mlp_1_h, mlp_1_h)

        # # Add dropout for mlp 1
        with tf.name_scope("dropout_mlp_1"):
            self.h_drop_mlp_1 = tf.nn.dropout(self.mlp_1_output, self.dropout_keep_prob)

        # MLP layer 2
        with tf.name_scope("mlp_2"):
            W = tf.Variable(tf.truncated_normal([int(num_filters_total / 2), int(num_filters_total / 4)], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0., shape=[int(num_filters_total / 4)]), name="b")
            mlp_2_h = tf.nn.xw_plus_b(self.h_drop_mlp_1, W, b, name="mlp_2_h")
            # mlp_2_h = tf.nn.xw_plus_b(self.mlp_1_output, W, b, name="mlp_2_h")
            # Apply nonlinearity
            # self.mlp_1_output = tf.nn.relu(mlp_1_h)
            self.mlp_2_output = tf.maximum(0.2 * mlp_2_h, mlp_2_h)

        # Add dropout for mlp 2
        with tf.name_scope("dropout_mlp_2"):
            self.h_drop_mlp_2 = tf.nn.dropout(self.mlp_2_output, self.dropout_keep_prob)

        # MLP layer 3
        with tf.name_scope("mlp_3"):
            W = tf.Variable(tf.truncated_normal([int(num_filters_total / 4), int(num_filters_total / 8)], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0., shape=[int(num_filters_total / 8)]), name="b")
            mlp_3_h = tf.nn.xw_plus_b(self.h_drop_mlp_2, W, b, name="mlp_3_h")
            # Apply nonlinearity
            # self.mlp_1_output = tf.nn.relu(mlp_1_h)
            self.mlp_3_output = tf.maximum(0.2 * mlp_3_h, mlp_3_h)

        # Add dropout for mlp 3
        with tf.name_scope("dropout_mlp_3"):
            self.h_drop_mlp_3 = tf.nn.dropout(self.mlp_3_output, self.dropout_keep_prob)

        # MLP layer 4
        with tf.name_scope("mlp_4"):
            W = tf.Variable(tf.truncated_normal([int(num_filters_total / 8), int(num_filters_total / 16)], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0., shape=[int(num_filters_total / 16)]), name="b")
            mlp_4_h = tf.nn.xw_plus_b(self.h_drop_mlp_3, W, b, name="mlp_4_h")
            # Apply nonlinearity
            # self.mlp_1_output = tf.nn.relu(mlp_1_h)
            self.mlp_4_output = tf.maximum(0.2 * mlp_4_h, mlp_4_h)

        # Add dropout for mlp 4
        with tf.name_scope("dropout_mlp_4"):
            self.h_drop_mlp_4 = tf.nn.dropout(self.mlp_4_output, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            W = tf.Variable(tf.truncated_normal([int(num_filters_total / 16), num_classes], stddev=0.1), name="W")
            # W = tf.Variable(tf.truncated_normal([int(num_filters_total/2), num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.2, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # self.scores = tf.nn.xw_plus_b(self.h_drop_conv, W, b, name="scores")
            self.scores = tf.nn.xw_plus_b(self.h_drop_mlp_4, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
