# Reference to http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

"""
Trains a CNN for text classification.

"""
import tensorflow as tf
import os
import time
from datetime import datetime
import numpy as np
from cnn_model import *
from corpus_loader123456789 import *

# set train & test file path
dict_path = 'dataOutput/sense_dict.txt'
train_path = 'dataOutput/sense_news_encoded/6_topic_news_train.txt'

NUM_CORES = 6


def get_batch(x, y, batch_size=batch_size):
    batch_index = 0
    nbatches = int(len(y) / batch_size)
    while batch_index < nbatches:
        batch_start = batch_index * batch_size
        batch_end = batch_start + batch_size
        batch_x = x[batch_start:batch_end]
        batch_y = y[batch_start:batch_end]
        batch_index += 1
        yield batch_x, batch_y


def log_settings(out_dir):
    print('Start to log the settings...')
    with open(out_dir + '/settings.txt', 'w', encoding='utf8') as log_out:
        for attr, value in sorted(FLAGS.__flags.items()):
            log_out.write("{}={}\n".format(attr, value))
        log_out.write('train_path={}\n'.format(train_path))


train_x, train_y, train_vocabulary, train_token_to_index = corpus_loader(corpus_path=train_path,
                                                                         dict_path=dict_path, vocab=None,
                                                                         max_sequence_length=max_sequence_length,
                                                                         random_seed=random_seed,
                                                                         ignore_short=True
                                                                         )
print("Train category stats:\n{}".format(category_stats_print(category_stats(train_y))))
# dev_x, dev_y, _, _ = corpus_loader(filename=eval_path, vocab=train_vocabulary,
#                                    max_sequence_length=max_sequence_length, random_seed=random_seed,
#                                    max_doc_count=5000)

# data augmentation n times
# permute_times = 10
# permute_train_x = np.zeros((train_x.shape[0] * permute_times, max_sequence_length), dtype=np.int16)
# permute_train_y = np.zeros((train_y.shape[0] * permute_times, num_classes), dtype=np.int16)
# np.random.seed(random_seed)
# for idx_y in range(train_y.shape[0]):
#     non_zero_or_padding = np.unique(train_x[idx_y][np.nonzero(train_x[idx_y] > 1)])
#     for augment_offset in range(permute_times):
#         permute_train_x[idx_y * permute_times + augment_offset][:len(non_zero_or_padding)] = np.random.permutation(non_zero_or_padding)
#         permute_train_y[idx_y * permute_times + augment_offset] = train_y[idx_y]
# print("Train docs: {}".format(len(permute_train_y)))
# np.random.seed(random_seed)
# train_x = np.random.permutation(permute_train_x)
# np.random.seed(random_seed)
# train_y = np.random.permutation(permute_train_y)

# training process
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        # inter_op_parallelism_threads=NUM_CORES,
        # intra_op_parallelism_threads=NUM_CORES
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNNModel(  # batch_size=batch_size,
                           sequence_length=max_sequence_length,
                           num_classes=num_classes,
                           vocab_size=len(train_vocabulary),
                           embedding_size=FLAGS.embedding_dim,
                           filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                           num_filters=FLAGS.num_filters,
                           l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)  # 1e-4 = 0.0001
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        # grad_summaries = []
        # for g, v in grads_and_vars:
        #     if g is not None:
        #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #         grad_summaries.append(grad_hist_summary)
        #         grad_summaries.append(sparsity_summary)
        # grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Log the model settings
        log_settings(out_dir)

        # Dev summaries
        # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(max_to_keep=50)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Assign pre-trained word2vec and template filter weight
        if FLAGS.word2vec_path:
            # initial matrix with random uniform
            initW = np.random.uniform(-0.25, 0.25, (len(train_vocabulary), FLAGS.embedding_dim))
            # set the '[__PAD__]' to 0.0
            initW[0] = [float(0)] * FLAGS.embedding_dim

            word2vec_dict = dict()  # word: vectors

            # load every vectors from the word2vec
            print('Load word2vec file "{}"'.format(FLAGS.word2vec_path))
            with open(FLAGS.word2vec_path, 'r', encoding='utf8') as vec_in:
                lines = [line.strip() for line in vec_in.readlines()]
            for line in lines:
                word, vectors = line.split()
                vectors = [float(x) for x in vectors.split(',')]
                initW[train_token_to_index[word]] = vectors
                word2vec_dict[word] = vectors
            print('Load word2vec completely!\nStart to assign initW to cnn.W')
            sess.run(cnn.W.assign(initW))
        # end of assign pre-trained word2vec


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {cnn.input_x: x_batch,
                         cnn.input_y: y_batch,
                         cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                                                          feed_dict)
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            if step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def eval_step(x_batch, y_batch, writer=None):
            """
            A single eval step
            """
            feed_dict = {cnn.input_x: x_batch,
                         cnn.input_y: y_batch,
                         cnn.dropout_keep_prob: 1.}
            step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                                                       feed_dict)
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        print(datetime.now().strftime('%a, %Y-%m-%d %H:%M:%S'))
        for ep in range(FLAGS.num_epochs):
            print("Epoch: {}".format(ep+1))
            # Generate batches
            train_batches = get_batch(train_x, train_y, FLAGS.batch_size)
            # Training loop. For each batch...
            for batch in train_batches:
                batch_x, batch_y = batch
                train_step(batch_x, batch_y)
                current_step = tf.train.global_step(sess, global_step)
                # if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
