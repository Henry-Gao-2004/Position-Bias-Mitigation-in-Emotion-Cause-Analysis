from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import tensorflow as tf
from tensorflow.keras import layers
from utils.tf_funcs import *
import layers as layers_lib

# Replace tf.app.flags with argparse or a custom configuration class
class FLAGS:
    max_grad_norm = 5.0
    learning_rate = 0.001
    generator_learning_rate = 0.001
    max_doc_len = 75
    max_sen_len = 30
    embedding_dim = 300
    embedding_dim_pos = 50
    n_hidden = 128
    n_class = 2
    lambda1 = 0.5
    l2_reg = 0.01
    batch_size = 32
    use_position = 'PAE'
    use_DGL = True
    hierachy = True
    scope = 'default'

def paedgl_optimize(loss, global_step=None):
    return layers_lib.adam_optimize(loss, global_step, FLAGS.learning_rate, FLAGS.max_grad_norm)

def adam_optimize(loss, global_step=None):
    return layers_lib.adam_optimize(loss, global_step, FLAGS.learning_rate, FLAGS.max_grad_norm)

def gene_adam_optimize(loss, global_step=None):
    return layers_lib.adam_optimize(loss, global_step, FLAGS.generator_learning_rate, FLAGS.max_grad_norm)

def optimize(loss, global_step=None):
    return layers_lib.optimize(
        loss, global_step, FLAGS.max_grad_norm, FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor)

def one2many_attention(emotion_pos, clause_reps):
    batch_idx = tf.expand_dims(tf.range(0, tf.shape(emotion_pos)[0]), 1)
    emotion_idx = tf.expand_dims(emotion_pos, 1)
    emotion_idx = tf.concat([batch_idx, emotion_idx], 1)
    emotion_tensor = tf.expand_dims(tf.gather_nd(clause_reps, emotion_idx), 1)  # [bs,1,out_units]
    scores = tf.squeeze(tf.matmul(emotion_tensor, clause_reps, transpose_b=True), 1)  # [?,75]
    attention = tf.nn.softmax(scores, axis=1)
    return attention

class peaModel(tf.Module):
    def __init__(self, cl_logits_input_dim=None):
        super(peaModel, self).__init__()
        self.layers = {}

    def get_doc_data(self, y_p_data, y_data, x_data, sen_len_data, doc_len_data, word_distance, DGL_data, test=False):
        for index in batch_index(len(y_data), FLAGS.batch_size, test):  # convey the real value
            feed_list = [x_data[index], sen_len_data[index], doc_len_data[index], y_data[index], y_p_data[index]]
            yield feed_list

    @tf.function
    def logits_extractor(self, word_embedding, pos_embedding, x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, RNN, scope_name):
        with tf.name_scope(scope_name):
            x = tf.nn.embedding_lookup(word_embedding, x)
            inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
            word_dis = tf.nn.embedding_lookup(pos_embedding, word_dis)
            sen_dis = word_dis[:, :, 0, :]
            word_dis = tf.reshape(word_dis, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim_pos])
            if FLAGS.use_position == 'PAE':
                inputs = tf.concat([inputs, word_dis], axis=2)
            inputs = tf.nn.dropout(inputs, rate=1 - keep_prob1)
            sen_len = tf.reshape(sen_len, [-1])
            inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope="word_layer" + scope_name)
            s = att_var(inputs, sen_len, get_weight_varible('word_att_w1' + scope_name, [2 * FLAGS.n_hidden, 2 * FLAGS.n_hidden]),
                        get_weight_varible('word_att_b1' + scope_name, [2 * FLAGS.n_hidden]),
                        get_weight_varible('word_att_w2' + scope_name, [2 * FLAGS.n_hidden, 1]))
            s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
            n_feature = 2 * FLAGS.n_hidden
            if FLAGS.use_position == 'PEC':
                s = tf.concat([s, sen_dis], axis=2)
                n_feature = 2 * FLAGS.n_hidden + FLAGS.embedding_dim_pos
            if FLAGS.hierachy:
                s = RNN(s, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_layer' + scope_name)
                n_feature = 2 * FLAGS.n_hidden
            s = tf.reshape(s, [-1, n_feature])
            s = tf.nn.dropout(s, rate=1 - keep_prob2)
            w_p = get_weight_varible('position_w' + scope_name, [n_feature, 102])
            b_p = get_weight_varible('position_b' + scope_name, [102])
            pred_p = tf.matmul(s, w_p) + b_p
            pred_p = tf.nn.softmax(pred_p)
            pred_p = tf.reshape(pred_p, [-1, FLAGS.max_doc_len, 102])
            w = get_weight_varible('cause_w' + scope_name, [n_feature, FLAGS.n_class])
            b = get_weight_varible('cause_b' + scope_name, [FLAGS.n_class])
            pred_c_tr = tf.matmul(s, w) + b
            pred_c_tr = tf.nn.softmax(pred_c_tr)
            pred_c_tr = tf.reshape(pred_c_tr, [-1, FLAGS.max_doc_len, FLAGS.n_class])
        return s, pred_c_tr, w, b, w_p, b_p, pred_p

    def get_batch_data(self, x, word_dis, DGL, label_pos, emotion_pos, sen_len, doc_len, keep_prob1, keep_prob2, y, y_p, batch_size, test=False):
        for index in batch_index(y.shape[0], FLAGS.batch_size, test=False):
            feed_list = [x[index], word_dis[index], DGL[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, y[index], y_p[index], label_pos[index], emotion_pos[index]]
            yield feed_list, len(index)

    @tf.function
    def get_original_prob(self, word_embedding, pos_embedding):
        x = tf.keras.Input(shape=(FLAGS.max_doc_len, FLAGS.max_sen_len), dtype=tf.int32, name='ori_x')
        word_dis = tf.keras.Input(shape=(FLAGS.max_doc_len, FLAGS.max_sen_len), dtype=tf.int32, name='ori_word_dis')
        DGL = tf.keras.Input(shape=(FLAGS.max_doc_len, FLAGS.max_doc_len), dtype=tf.float32, name='ori_DGL')
        sen_len = tf.keras.Input(shape=(FLAGS.max_doc_len,), dtype=tf.int32, name='ori_sen_len')
        doc_len = tf.keras.Input(shape=(), dtype=tf.int32, name='ori_doc_len')
        keep_prob1 = tf.keras.Input(shape=(), dtype=tf.float32, name="ori_keep_prob1")
        keep_prob2 = tf.keras.Input(shape=(), dtype=tf.float32, name="ori_keep_prob2")
        y = tf.keras.Input(shape=(FLAGS.max_doc_len, FLAGS.n_class), dtype=tf.int32, name='ori_label')
        y_p = tf.keras.Input(shape=(FLAGS.max_doc_len, 102), dtype=tf.float32, name='ori_p_label')

        s_tr, pred_c_tr, w, b, w_p, b_p, pred_p = self.logits_extractor(word_embedding, pos_embedding, x, word_dis, DGL, sen_len, doc_len, keep_prob1, keep_prob2, RNN=biLSTM, scope_name="original")
        logits = pred_c_tr
        softmax_prob = tf.reshape(logits, [-1, logits.shape[-1]])
        new_y = tf.cast(tf.argmax(y, axis=2), tf.int32)
        original_prob = tf.gather_nd(softmax_prob, tf.stack((tf.range(tf.shape(softmax_prob)[0], dtype=tf.int32), tf.reshape(new_y, [-1])), axis=1))
        return original_prob
