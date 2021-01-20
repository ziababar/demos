import os
import sys
import getopt
import filetype
import random
import string
import magic
import time
import zlib
from os import listdir
from os.path import isfile, join

# import the required libraries
import gensim
import nltk
import numpy
import requests
import cv2
import matplotlib.pyplot as plt 
import numpy as np

import bs4 as bs
import urllib.request
import re

import pymongo


from urllib.request import urlopen


def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

def crc32(fileName):
    """Compute the CRC-32 checksum of the contents of the given filename"""
    with open(fileName, "rb") as f:
        checksum = 0
        while (chunk := f.read(65536)) :
            checksum = zlib.crc32(chunk, checksum)
        return checksum

def check_filetype(fileName):
    fileType = magic.from_file(fileName)
    return(fileType.split(' ', 1)[0])


def vectorize_text(fileName):

    source_doc = fileName
    print(source_doc)

    f = open(fileName, "r")
    article_text = f.read()

    corpus = nltk.sent_tokenize(article_text)

    for i in range(len(corpus )):
        corpus [i] = corpus [i].lower()
        corpus [i] = re.sub(r'\W',' ',corpus [i])
        corpus [i] = re.sub(r'\s+',' ',corpus [i])

    wordfreq = {}
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    import heapq
    most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

    word_idf_values = {}
    for token in most_freq:
        doc_containing_word = 0
        for document in corpus:
            if token in nltk.word_tokenize(document):
                doc_containing_word += 1
        word_idf_values[token] = np.log(len(corpus)/(1 + doc_containing_word))

    word_tf_values = {}
    for token in most_freq:
        sent_tf_vector = []
        for document in corpus:
            doc_freq = 0
            for word in nltk.word_tokenize(document):
                if token == word:
                    doc_freq += 1
            word_tf = doc_freq/len(nltk.word_tokenize(document))
            sent_tf_vector.append(word_tf)
        word_tf_values[token] = sent_tf_vector

    tfidf_values = []
    for token in word_tf_values.keys():
        tfidf_sentences = []
        for tf_sentence in word_tf_values[token]:
            tf_idf_score = tf_sentence * word_idf_values[token]
            tfidf_sentences.append(tf_idf_score)
        tfidf_values.append(tfidf_sentences)

    tf_idf_model = np.asarray(tfidf_values)

    tf_idf_model = np.transpose(tf_idf_model)
    tf_idf_model_string = tf_idf_model.tolist()

    fileRecord = { "fileType": "TEXT", "fileCrc": crc32(fileName), "fileSize": os.stat(fileName).st_size, "fileName": fileName, "fileData": tf_idf_model_string }

    return fileRecord



def vectorize_image(fileName):

    url = "https://raw.githubusercontent.com/ziababar/demos/master/derivative_security/data/original_golden_bridge-300x169.jpg"

    # download the image, convert it to a NumPy array, and then read
    resp = urlopen(url)
    # image = np.asarray(bytearray(fileName.read()), dtype="uint8")
    image = fileName
    image = cv2.imread(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Construct a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # Detect key points and descriptors both both images
    kp, desc = sift.detectAndCompute(image, None)

    return


if __name__ == '__main__':

    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["torontodb"]
    myCollection = mydb["fileList"]


    print(myclient.list_database_names())

    folder = ""
    filesList = []
    oldFilesList = []
    
    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "hf:"
    
    # Long options
    long_options = ["Help", "Folder ="]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:
    
            if currentArgument in ("-h", "--Help"):
                print ("Usage: agent.py -f <folder>")
                               
            elif currentArgument in ("-f", "--Folder"):
                folder = currentValue
                # print (("Enabling special output mode (% s)") % (currentValue))
                print(("Input folder is %s") % (folder))
                
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    # Loop continously checking the folder
    i = 1
    while True:

        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        filesList = Diff(files, oldFilesList)
        print("List of files are ", filesList)

        for fileName in filesList:
            fileRecord = ""
            fileType = check_filetype(folder+fileName)

            if fileType == "ASCII":
                fileRecord = vectorize_text(folder+fileName)
                x = myCollection.insert_one(fileRecord)
                print(fileRecord)
            elif fileType == "JPEG":
                print("File type: JPEG")
                vectorize_image(folder+fileName)
            else:
                print("Unknown file type, skipping...")

        oldFilesList = filesList

        i += 1

        if i > 2:
            sys.exit(2)

        time.sleep(10)


