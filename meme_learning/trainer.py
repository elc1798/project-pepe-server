from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
tf.logging.set_verbosity(tf.logging.INFO)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import namedtuple
import glob

import preprocessing

"""
Meme classification model:

    Preprocess ==> Resize images to 299x299, extract probable objects

    Inputs:
        image (299x299), main_objects (5x2), column 1 - ids, column 2 - ratings

        Note: After all data as been accumulated, we will one-hot the
        main_object vectors based on ids, and pad with 0s to fill in missing
        things

        Each input (image + main_objects) will have its own "dankness" rating,
        an from 1 - 10 inclusive.

    Operations:
        1. Pad each image into an input layer [-1, 300, 300, 1]

        2. Convolutional Layer using tf.layers.conv2d, size 32 kernel:
            [-1, 300, 300, 32]

        3. Pooling Layer with Max Pooling, size 2x2 pool, stride of 2:
            [-1, 150, 150, 1]

        4. 2nd Convolutional Layer using conv2d, size 64 kernel:
            [-1, 150, 150, 64]

        5. 2nd Pooling Layer with Max Pooling, size 2x2 pool, stride of 2:
            [-1, 75, 75, 64]

        6. Neuron Dense Layer for image feature extraction:
            Flatten into [-1, 75x75x64]
            Use 2^10 units for dense layer, using ReLU as our activation
            Add dropout regularization

            Outputs [-1, 1024]

        7. Logit Layer with 10 units (each denoting a rating between 1 - 10)
"""

A, y = None, None

npyfs = glob.glob("*.npy")
if "mat_A.npy" in npyfs and "mat_y.npy" in npyfs:
    print("USING PRELOADED TRAINING SET")
    A = np.load("mat_A.npy")
    y = np.load("mat_y.npy")
else:
    A, y = preprocessing.get_training_set()
    np.save("mat_A.npy", A)
    np.save("mat_y.npy", y)

print(A.dtype, y.dtype)
print(A.shape)

memes_classifier = None

# A should be (n, 4k, 4r)
MATRIX_SHAPE = A.shape

def meme_model(features, labels, mode):

    input_layer = tf.reshape(features, [-1, MATRIX_SHAPE[1], MATRIX_SHAPE[2], 1])

    # Convolutional Layer #1
    # Input Tensor Shape: [batch_size, x, y, 1]
    # Output Tensor Shape: [batch_size, x, y, 32]
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32,
                             kernel_size=[5, 5], padding='same',
                             activation=tf.nn.relu)

    # Pooling Layer #1
    # Input Tensor Shape: [batch_size, x, y, 32]
    # Output Tensor Shape: [batch_size, x / 2, y / 2, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                    strides=2)

    # Convolutional Layer #2
    # Input Tensor Shape: [batch_size, x / 2, y / 2, 32]
    # Output Tensor Shape: [batch_size, x / 2, y / 2, 64]
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,
                             5], padding='same', activation=tf.nn.relu)

    # Pooling Layer #2
    # Input Tensor Shape: [batch_size, x / 2, y / 2, 64]
    # Output Tensor Shape: [batch_size, x / 4, y / 4, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                    strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, x / 4, y / 4, 64]
    # Output Tensor Shape: [batch_size, x/4 * y/4 * 64]
    pool2_flat = tf.reshape(pool2, [-1, MATRIX_SHAPE[1] * MATRIX_SHAPE[2] * 4])

    # Dense Layer
    # Input Tensor Shape: [batch_size, x * y * 4]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode
                                == learn.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(
            indices=tf.cast(labels, tf.int32),
            depth=10
        )

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits
        )

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.0001,
            optimizer='SGD'
        )

    # Generate Predictions
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )

def train(_):
    global A, y, memes_classifier

    # Load training and eval data
    train_data = A
    train_labels = y
    eval_data = A[:25].copy()
    eval_labels = y[:25].copy()

    # Create the Estimator
    if memes_classifier == None:
        memes_classifier = learn.Estimator(model_fn=meme_model,
            model_dir='models/')

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {'probabilities': 'softmax_tensor'}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
    #         every_n_iter=50)

    # Train the model
    memes_classifier.fit(x=train_data, y=train_labels, batch_size=100,
        # steps=4000, monitors=[logging_hook])
        steps=40)

    # Configure the accuracy metric for evaluation
    metrics = {
        'accuracy': learn.MetricSpec(
            metric_fn=tf.metrics.accuracy,
            prediction_key='classes'
        )
    }

    # Evaluate the model and print results
    eval_results = memes_classifier.evaluate(
        x=eval_data,
        y=eval_labels,
        metrics=metrics
    )

    print(eval_results)

def get_rating(parsed_img):
    global memes_classifier

    if memes_classifier == None:
        memes_classifier = learn.Estimator(model_fn=meme_model,
            model_dir='models/')

    res = memes_classifier.predict(x=parsed_img)
    print(res)
    for i in res:
        print(i)

if __name__ == "__main__":
    # tf.app.run(main=train)
    get_rating(A[100])
    print(y[100])

