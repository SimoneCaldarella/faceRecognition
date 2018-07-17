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

    subprocess.Popen(['/bin/bash', '-c', cmd]).wait()

###############-First part, used to get images for the training-################

def facialRecognition():

    '''This is the main function for the facial recognition
    and images cropping'''

    name = input("Insert your name: ").strip()
    dirName = input("Insert the directory where you want to save your images(Default ./images/yourName): ").strip()
    if (dirName == ""):
        dirName = os.path.join(".", "images")
        os.mkdir(dirName)
    dirName = os.path.join(dirName, name)
    try:
        os.mkdir(os.path.join(".", dirName))
    except:
        print("Folder already exists")
    bash_command("python3 saveFaces.py " + dirName + " " + name)

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

def makeInference():

    '''Main function to make inference'''

    img_dir = input("Insert the path of the image: ").strip()

    graph_path = input("Insert the graph's path(Default = ./outputGraph/output_graph.pb): ").strip()
    if (graph_path == ""):
        graph_path = os.path.join(".", "outputGraph", "output_graph.pb")

    labels_path = input("Insert the label's path(Default = ./labels/output_labels.txt): ").strip()
    if (labels_path == ""):
        labels_path = os.path.join(".", "labels", "output_labels.txt")

    bash_command("python3 inference.py " + img_dir + " " + graph_path + " " + labels_path)

def showMenu():

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
            bash_command("clear")


        elif (choice == '2'):
            training()
            bash_command("clear")

        elif (choice == '3'):
            makeInference()
            bash_command("clear")

        elif (choice == '4'):
            exit(0)

        else:
            print("Please select an existing choice!")
            bash_command("clear")
