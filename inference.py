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

import tensorflow as tf

def check_version():
    v = tf.__version__
    if not v.startswith("2.19"):
        print(f"Warning: Running under tf.__version__ = {v}, but 2.19.x is recommended.")
    tf.compat.v1.disable_v2_behavior()  # enables tf1.x graph + Session API

# first, set flags (must be defined *before* importing ad_graph or utils.prepare_data)
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'data/train_ckpt', 'path to checkpoint directory')
flags.DEFINE_string('model_type', 'peadgl', '')
flags.DEFINE_string('mode', 'test', 'ignored')
flags.DEFINE_integer('embedding_dim', 200, '')
flags.DEFINE_integer('embedding_dim_pos', 50, '')
flags.DEFINE_integer('max_sen_len', 30, '')
flags.DEFINE_integer('max_doc_len', 75, '')
flags.DEFINE_integer('n_hidden', 100, '')
flags.DEFINE_integer('n_class', 2, '')
flags.DEFINE_float('lambda1', 0.1, '')
flags.DEFINE_float('learning_rate', 0.005, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('dis_warm_up_step', 50, '')
flags.DEFINE_integer('gene_warm_up_step', 50, '')
flags.DEFINE_integer('random_seed', 29, '')
flags.DEFINE_integer('max_steps', 100000, '')
flags.DEFINE_string('w2v_file', 'data/w2v_200.txt', '')

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
    check_version()

    embedding_dim = 300
    embedding_dim_pos = 50
    word2idx, embeddings, embedding_pos = load_w2v(embedding_dim, embedding_dim_pos, "data/clause_keywords.csv", "data/w2v_200.txt")

    max_sen_len = FLAGS.max_sen_len
    max_doc_len = FLAGS.max_doc_len

    sentence = "I am happy that the program ran"
    doc_ids = [preprocess_one_sentence(sentence, word2idx, max_sen_len)]
    x_in, wdis_in, DGL_in, slen_in, dlen_in = build_inputs(doc_ids, max_doc_len, max_sen_len)

    # build the model structure
    model = ad_graph.peaModel()
    word_emb_var = tf.constant(embeddings, dtype=tf.float32, name='word_embedding')
    pos_emb_var = tf.constant(
        embeddings[:, :FLAGS.embedding_dim_pos], dtype=tf.float32, name='pos_embedding')

    x_ph = tf.compat.v1.placeholder(tf.int32, [None, max_doc_len, max_sen_len], name='x')
    word_dis_ph = tf.compat.v1.placeholder(tf.int32, [None, max_doc_len, max_sen_len], name='word_dis')
    DGL_ph = tf.compat.v1.placeholder(tf.float32, [None, max_doc_len, max_doc_len], name='DGL')
    sen_len_ph = tf.compat.v1.placeholder(tf.int32, [None, max_doc_len], name='sen_len')
    doc_len_ph = tf.compat.v1.placeholder(tf.int32, [None], name='doc_len')
    keep1 = tf.compat.v1.placeholder(tf.float32, [])
    keep2 = tf.compat.v1.placeholder(tf.float32, [])
    y_dummy = tf.compat.v1.placeholder(tf.float32, [None, max_doc_len, FLAGS.n_class], name='y')
    y_p_dummy = tf.compat.v1.placeholder(tf.float32, [None, max_doc_len, 102], name='y_p')

    pred_c_tr, pred_c_te, pred_p, reg_loss = model.build_model(
        word_emb_var, pos_emb_var,
        x_ph, word_dis_ph, DGL_ph,
        sen_len_ph, doc_len_ph, keep1, keep2)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {FLAGS.train_dir}")
        saver.restore(sess, ckpt)

        probs = sess.run(pred_c_te, feed_dict={
            x_ph: x_in, word_dis_ph: wdis_in, DGL_ph: DGL_in,
            sen_len_ph: slen_in, doc_len_ph: dlen_in,
            keep1: 1.0, keep2: 1.0
        })

    p0, p1 = probs[0, 0]  # [non‑cause, cause]
    label = 'CAUSE' if p1 > p0 else 'NON‑CAUSE'
    print("Sentence:", sentence)
    print(f"Scores → non‑cause: {p0:.4f}, cause: {p1:.4f}")
    print("Predicted clause label:", label)

if __name__ == "__main__":
    main()
