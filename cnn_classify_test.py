
"""
Test a CNN for text classification.

"""
import tensorflow as tf
import numpy as np
import os
import sys
import glob
from datetime import datetime
from colorama import Fore, Back, Style
from cnn_model import *
from cnn_corpus_loader import *

# set train & test file path
dict_path = 'data/word2vec/word_dict.txt'
valid_path = 'dataOutput/validation/labeledTrainData_clean_validation_encoded.tsv'


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


valid_x, valid_y, train_vocabulary, _ = cnn_corpus_loader(corpus_path=valid_path,
                                                          dict_path=dict_path, vocab=None,
                                                          max_sequence_length=max_sequence_length,
                                                          random_seed=random_seed, max_doc_count=max_doc_count,
                                                          ignore_short=False)
print("Test docs: {}".format(len(valid_y)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNNModel(sequence_length=max_sequence_length,
                           num_classes=num_classes,
                           vocab_size=len(train_vocabulary),
                           embedding_size=FLAGS.embedding_dim,
                           filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                           num_each_filter=FLAGS.num_filters,
                           l2_reg_lambda=FLAGS.l2_reg_lambda)

        saver = tf.train.Saver()

        # load saved model
        all_dir = os.listdir('./runs/')
        last_run = sorted([dir for dir in all_dir if len(dir) == 10 and dir.isdigit()])[-1]
        # all model with data
        model_names = [fm for fm in glob.glob('./runs/' + last_run + '/checkpoints/model-*') if '.data' in fm]
        model_steps = list()
        for model in model_names:
            prefix, step_count = model[:model.index('.data')].split('model-')
            model_steps.append(int(step_count))

        model_steps = sorted(model_steps)
        prefix += 'model-'

        last_model_steps = str(model_steps[-1])
        output_filename = 'performance_' + last_model_steps + '.txt'
        with open('./runs/' + last_run + '/' + output_filename, 'w', encoding='utf8') as result_out:
            for step in model_steps:  # cancel reversed(model_steps)
                model = prefix + str(step)
                saver.restore(sess, model)
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print('Stat to test this model, {}'.format(time_str))
                print('Testing with run {} model {}'.format(last_run, step))
                result_out.write('Testing with run {} model {}\n'.format(last_run, step))
                sum_table = np.zeros((3, num_classes))
                predict_batches = get_batch(valid_x, valid_y)
                batch_num = 0
                for pbatch in predict_batches:
                    print("\rTesting batch {:>5}".format(batch_num + 1), end='')
                    sys.stdout.flush()
                    batch_num += 1
                    batch_x, batch_y = pbatch
                    answers = np.argmax(batch_y, axis=1)
                    feed_dict = {cnn.input_x: batch_x,
                                 cnn.input_y: batch_y,
                                 cnn.dropout_keep_prob: 1.0
                                }
                    output = sess.run([cnn.predictions], feed_dict)
                    predicted = output[0]
                    for idx in range(len(answers)):  # batch_sizes
                        sum_table[1, predicted[idx]] += 1  # TP + FP
                        sum_table[2, answers[idx]] += 1  # TP + FN
                        if predicted[idx] == answers[idx]:
                            sum_table[0, answers[idx]] += 1  # TP
                performances = np.zeros((3, num_classes))
                performances[0, :] = sum_table[0, :] / (sum_table[1, :] + 1e-7)  # precision
                performances[1, :] = sum_table[0, :] / (sum_table[2, :] + 1e-7)  # recall
                performances[2, :] = (2 * performances[0, :] * performances[1, :]) / (performances[0, :] + performances[1, :] + 1e-7)  # F-score
                print('')
                for topic_index in range(num_classes):
                    print('{precision:.4f}\t{recall:.4f}\t{f1:.4f}'.format(precision=performances[0, topic_index],
                                                                           recall=performances[1, topic_index],
                                                                           f1=performances[2, topic_index]
                                                                           ))
                    result_out.write('{precision:.4f}\t{recall:.4f}\t{f1:.4f}\n'.format(precision=performances[0, topic_index],
                                                                                        recall=performances[1, topic_index],
                                                                                        f1=performances[2, topic_index]))
                total = np.sum(sum_table, axis=1)  # [TP, TP + FP, TP + FN]
                micro_precision = total[0] / (total[1] + 1e-7)
                micro_recall = total[0] / (total[2] + 1e-7)
                micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
                print('Micro Average F-score {}{:.4f}{}'.format(Back.GREEN, micro_fscore, Style.RESET_ALL))
                print('Macro Average F-score {}{:.4f}{}'.format(Back.GREEN, np.mean(performances[2]), Style.RESET_ALL))
                result_out.write('Micro Average F-score {f1:.4f}\n'.format(f1=micro_fscore))
                result_out.write('Macro Average F-score {f1:.4f}\n\n'.format(f1=np.mean(performances[2])))
                result_out.flush()
