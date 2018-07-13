from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Simone Caldarella"
__copyright__ = "Copyright (C) 2018 Simone Caldarella"
__license__ = "Apache 2.0"
__version__ = "1.0"

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

def bash_command(cmd):

    '''Best way to run shell command from python'''

    subprocess.Popen(['/bin/bash', '-c', cmd])

###############-First part, used to get images for the training-################

def facialLandMarksRecognition(dirName, name):

    '''In this function the algorithm recognize your
    face and save every frames of the video keeping only the face'''

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cam = cv2.VideoCapture(0)
    timy = 6
    cont = 0
    sec = (int(round(time.time())))
    deltaSec = (int(round(time.time()))) - sec

    while (deltaSec < timy):

        deltaSec = (int(round(time.time()))) - sec
        val, image = cam.read()
        image = imutils.resize(image, width=700)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        title = name+str(cont)+'.jpg'
        dir = os.path.join(dirName, title)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w + 20, y + h +20), (0, 255, 0), 2)
            img = image[y:y+h+10, x:x+w+10]
            cv2.imwrite(dir, img)
            cont = cont + 1
            cv2.putText(image, "Face #{} {}".format(i + 1, deltaSec), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (x, y) in shape:
        	    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow("Output", image)
        cv2.waitKey(1)

    cv2.waitKey(1)
    cv2.destroyAllWindows()

def facialRecognition():

    '''This is the main function for the facial recognition
    and images cropping'''

    name = input("Insert your name: ").strip()
    dirName = input("Insert the directory where you want to save your images(Default ./images/yourName): ").strip()
    if (dirName == ""):
        dirName = os.path.join(".", "images")
        os.mkdir(dirName)
    dirName = os.path.join(dirName, name)
    os.mkdir(os.path.join(".", dirName))
    facialLandMarksRecognition(dirName, name)

###############-Second part, used to make the training-##################

def training():

    '''This is the main function for the training'''

    image_dir = input("Insert the directory path of images you want to use for the training: ").strip()
    image_dir = " --image_dir "+image_dir

    graph_dir = input("Insert the path where you want to save the output graph (+ name of the graph .pb)(Default = ./outputGraph/output_graph.pb): ").strip()
    if (graph_dir == ""):
        os.mkdir(os.path.join(".","outputGraph"))
        graph_dir = os.path.join(".", "outputGraph", "output_graph.pb")
    graph_dir = " --output_graph "+graph_dir

    labels_dir = input("Insert the path where you want to save the output labels (+ name of the labels .txt)(Default = ./labels/output_labels.txt): ").strip()
    if (labels_dir == ""):
        os.mkdir(os.path.join(".", "labels"))
        labels_dir = os.path.join(".", "labels", "output_labels.txt")
    labels_dir = " --output_labels "+labels_dir

    bottleneck_dir = input("insert the path where you want to save bottlenecks(Default = ./bottlenecks: ").strip()
    if (bottleneck_dir == ""):
        os.mkdir(os.path.join(".", "bottlenecks"))
        graph_dir = os.path.join(".", "bottlenecks")
    bottleneck_dir = " --bottleneck_dir "+bottleneck_dir

    bash_command("python3 retrain.py"+image_dir+graph_dir+labels_dir+bottleneck_dir)
    print("Training terminated!")

###############-Third part, used to make inference-################

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

def findMax(labels, results):

    '''It provides to find the most probable label'''

    index, value = max(enumerate(results), key=operator.itemgetter(1))
    return labels[index], value

def makeInference():

    '''Main function to make inference'''

    img_dir = input("Insert the path of the image: ").strip()

    graph_path = input("Insert the graph's path(Default = ./outputGraph/output_graph.pb): ").strip()
    if (graph_path == ""):
        graph_dir = os.path.join(".", "outputGraph", "output_graph.pb")

    labels_path = input("Insert the label's path(Default = ./labels/output_labels.txt): ").strip()
    if (graph_path == ""):
        graph_dir = os.path.join(".", "labels", "output_labels.txt")

    img = cv2.imread(img_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    while (True):
        if cv2.waitKey(0)==113:
            break
    cv2.destroyWindow("Inference")

def showMenu():

    os.bash_command("clear")

    print("@-----------------------------@")
    print("| 1) Prepare the dataset      |")
    print("| 2) Start the training       |")
    print("| 3) Make inference           |")
    print("| 4) Exit                     |")
    print("|-----------------------------|")
    print("| Author: Simone Caldarella   |")
    print("@-----------------------------@")
    print("")

if __name__ == "__main__":

    while (True):

        showMenu()

        choice = input("Insert the number of the task: ").strip()

        if (choice == '1'):
            facialRecognition()
            cv2.destroyAllWindows()

        elif (choice == '2'):
            training()

        elif (choice == '3'):
            makeInference()

        elif (choice == '4'):
            exit(0)

        else:
            print("Please select an existing choice!")
