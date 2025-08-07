#!/usr/bin/env python3
# Usage: python infer_paedgl_tf2.py
#
# ⚠️ This script runs in TF‑2.19 (or any 2.x version that still supports tf.compat.v1).
# It emulates a TF‑1.9 environment via tf.compat.v1.disable_v2_behavior()
#
# Place this script in the repo root (where ad_graph.py and utils/prepare_data.py are).
# Make sure your checkpoint dir (with .ckpt files) is set in train_dir below.

import os
import sys
import numpy as np
from ad_paedgl import load_data

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', type=str, default='./data/train_ckpt', help='Directory for logs and checkpoints.')
#>>>>>>>>>>>>>>>>>>>>> For models <<<<<<<<<<<<<<<<<<<<<<#
parser.add_argument('--model_type', type=str, default='peadgl', help='embedding file')
parser.add_argument('--mode', type=str, default='train_adv', help='embedding file')
# >>>>>>>>>>>>>>>>>>>> For peaModel <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
parser.add_argument('--w2v_file', type=str, default='data/w2v_200.txt', help='embedding file')
parser.add_argument('--embedding_dim', type=int, default=200, help='dimension of word embedding')
parser.add_argument('--embedding_dim_pos', type=int, default=50, help='dimension of position embedding')
parser.add_argument('--pos_trainable', type=str, default='', help='whether position embedding is trainable')
## input struct ##
parser.add_argument('--max_sen_len', type=int, default=30, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len', type=int, default=75, help='max number of sentences per documents')
## model struct ##
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--use_position', type=str, default='PAE', help='PAE or PEC')
parser.add_argument('--use_DGL', type=str, default='use', help='whether use DGL')
parser.add_argument('--hierachy', type=str, default='', help='whether use hierachy')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
parser.add_argument('--train_file_path', type=str, default='./data/clause_keywords.csv', help='training file')
parser.add_argument('--log_file_name', type=str, default='PAEDGL.log', help='name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
parser.add_argument('--training_iter', type=int, default=20, help='number of train iter')
parser.add_argument('--scope', type=str, default='PAEDGL', help='RNN scope')
parser.add_argument('--test_steps', type=int, default=200, help='test at every step')
parser.add_argument('--train_steps', type=int, default=20, help='show statistics of training')
parser.add_argument('--every', type=int, default=1, help='one sample generate 2 negative sample')  # number of batches negtive samples
# not easy to tune 
parser.add_argument('--batch_size', type=int, default=32, help='number of samples per batch')
parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
parser.add_argument('--keep_prob1', type=float, default=0.5, help='word embedding dropout keep prob')
parser.add_argument('--keep_prob2', type=float, default=1.0, help='softmax layer dropout keep prob')
parser.add_argument('--l2_reg', type=float, default=0.00001, help='l2 regularization rate')
parser.add_argument('--lambda1', type=float, default=0.1, help='rate for position prediction loss')

parser.add_argument('--dis_warm_up_step', type=int, default=50, help='discriminator warm up step')
parser.add_argument('--gene_warm_up_step', type=int, default=50, help='generator warm up step')

#>>>>>>>>>>>>>>>>>>For generator <<<<<<<<<<<<<<<<<<<#
parser.add_argument('--generator_learning_rate', type=float, default=0.001, help='rate for position prediction loss')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Clip the global gradient norm to this value.')

parser.add_argument('--random_seed', type=int, default=29, help='random_seed')
parser.add_argument('--max_steps', type=int, default=100000, help='random_seed')

FLAGS, unparsed = parser.parse_known_args()

# now import actual model and data utilities
import ad_graph  # defines peaModel class
from utils.prepare_data import load_w2v

def preprocess_one_sentence(sentence, word2idx, max_sen_len):
    tokens = sentence.strip().split()
    ids = [word2idx.get(tok, word2idx.get('<UNK>', 0)) for tok in tokens]
    if len(ids) > max_sen_len:
        ids = ids[:max_sen_len]
    pad = [word2idx.get('<PAD>', 0)] * (max_sen_len - len(ids))
    return ids + pad

def build_inputs(doc_ids, max_doc_len, max_sen_len):
    doc_len = len(doc_ids)
    word_dis = [[0] * max_sen_len for _ in range(doc_len)]
    DGL = [[1.0 if i == j else 0.0 for j in range(doc_len)] for i in range(doc_len)]
    sen_len = [sum(1 for w in sen if w != 0) for sen in doc_ids]

    x_np = np.zeros((1, max_doc_len, max_sen_len), dtype=np.int32)
    word_dis_np = np.zeros_like(x_np)
    DGL_np = np.zeros((1, max_doc_len, max_doc_len), dtype=np.float32)
    sen_len_np = np.zeros((1, max_doc_len), dtype=np.int32)
    doc_len_np = np.array([doc_len], dtype=np.int32)

    x_np[0, :doc_len] = doc_ids
    word_dis_np[0, :doc_len] = word_dis
    DGL_np[0, :doc_len, :doc_len] = DGL
    sen_len_np[0, :doc_len] = sen_len

    return x_np, word_dis_np, DGL_np, sen_len_np, doc_len_np

def main():
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)

    # build the model structure
    model = ad_graph.peaModel()
    embedding_dim = 200
    embedding_dim_pos = 50
    
    word_id_mapping, word_embedding, pos_embedding = load_w2v(embedding_dim, embedding_dim_pos, "data/clause_keywords.csv", "data/w2v_200.txt")

    y_p_data, y_data, x_data, sen_len_data, doc_len_data, word_distance, DGL_data, label_pos_data, emotion_pos_data = load_data(FLAGS.train_file_path, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)

    train_doc = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    train_word_dis = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    DGL = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.max_doc_len])
    train_sen_len = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    train_doc_len = tf.compat.v1.placeholder(tf.int32, [None])
    train_keep_prob1 = tf.compat.v1.placeholder(tf.float32)
    train_keep_prob2 = tf.compat.v1.placeholder(tf.float32)
    train_label = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    train_label_p = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.max_doc_len, 102])
    train_label_pos_op = tf.compat.v1.placeholder(tf.int32, [None])
    train_emotion_pos_op = tf.compat.v1.placeholder(tf.int32,[None])

    placeholders = [train_doc, train_word_dis, DGL, train_sen_len, train_doc_len, train_keep_prob1, train_keep_prob2, train_label, train_label_p, train_label_pos_op, train_emotion_pos_op]
    
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.Variable(pos_embedding, dtype=tf.float32, name='pos_embedding')
    dis_loss_op, train_dis_op, reward_op, pred_y, gt = model.train_discriminator(global_step,word_embedding, pos_embedding)

    tf.compat.v1.disable_eager_execution() 
    with tf.compat.v1.Session() as sess:
        tf.random.set_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        tf.compat.v1.global_variables_initializer().run(session=sess)

        ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
        print("Model restored from checkpoint:", ckpt)
        saver = tf.compat.v1.train.import_meta_graph('./data/train_ckpt/'+ckpt+'.meta')
        saver.restore(sess, ckpt)

        sentence = "I am happy because the program ran"
        doc_ids = [preprocess_one_sentence(sentence, word_id_mapping, FLAGS.max_sen_len)]
        x_np, word_dis_np, DGL_np, sen_len_np, doc_len_np = build_inputs(doc_ids, FLAGS.max_doc_len, FLAGS.max_sen_len)

        feed_dict = {
            train_doc: x_np,
            train_word_dis: word_dis_np,
            DGL: DGL_np,
            train_sen_len: sen_len_np,
            train_doc_len: doc_len_np,
            train_keep_prob1: 1.0,
            train_keep_prob2: 1.0
        }

        probs = sess.run(pred_y, feed_dict=feed_dict)

    p0, p1 = probs[0, 0]  # [non‑cause, cause]
    label = 'CAUSE' if p1 > p0 else 'NON‑CAUSE'
    print("Sentence:", sentence)
    print(f"Scores → non‑cause: {p0:.4f}, cause: {p1:.4f}")
    print("Predicted clause label:", label)

if __name__ == "__main__":
    main()
