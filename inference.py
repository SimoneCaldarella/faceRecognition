from imutils import face_utils

import cv2
import dlib
import imutils
import numpy as np
import operator
import os
import subprocess
import sys
import tensorflow as tf
import time
import tkinter as tk

def findMax(labels, results):

    '''It provides to find the most probable label'''

    index, value = max(enumerate(results), key=operator.itemgetter(1))
    return labels[index], value

def load_graph(model_file):

    '''Function used to load an already saved graph'''

    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):

    '''Convert the image in a tensor'''

    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):

    '''This function provides to load the lables into the graph'''

    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())

    return label

def inference(file_path, model_path, labels_path):

    '''This function is the core for the inference'''

    file_name = file_path
    model_file = model_path
    label_file = labels_path
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    return labels, results

def separate(word, wordArr):
    cont = 0
    sepWord = []
    for _ in word:
        sepWord.append(_)
        if (_ == " "):
            cont += 1
        if (cont == 2):
            wordArr.append(''.join(sepWord))
            sepWord = []
            cont = 0    

def imageClassification(img_dir, graph_path, labels_path):

    img = cv2.imread(img_dir)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        title = 'cropped'+str(i)+'.png'
        image = img[y:y+h, x:x+w]
        cv2.imwrite(title, image)
        title_path = title
        labels, results = inference(title_path, graph_path, labels_path)
        labels, results = findMax(labels, results)
        text = labels + " " + str(results)
        img = cv2.rectangle(img, (x-20, y-20), (x + w + 20, y + h +20), (0, 255, 0), 2)
        cv2.putText(img, text, (x-20, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Inference', img)
    cv2.waitKey(0)
    cv2.destroyWindow("Inference")

if __name__ == "__main__":

    img_dir = sys.argv[1]
    graph_path = sys.argv[2]
    labels_path = sys.argv[3]

    imageClassification(img_dir, graph_path, labels_path)
