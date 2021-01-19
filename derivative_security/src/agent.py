import sys
import getopt
import filetype
import magic
import time
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


from nltk.tokenize import word_tokenize # Word Tokenizer
from nltk.tokenize import sent_tokenize # Sentence Tokenizer
from urllib.request import urlopen


def check_filetype(fileName):
    fileType = magic.from_file(fileName)
    return(fileType.split(' ', 1)[0])


def vectorize_text(fileName):

    source_doc = fileName
    print(source_doc)

    # Empty array that contains all the sentences
    sent_array = []

    sent_tokens = sent_tokenize(source_doc)
    for line in sent_tokens:
        sent_array.append(line)

    print("Number of sentences: ", len(sent_array))
    print(sent_array)

    word_array = [[w.lower() for w in word_tokenize(text)] 
                for text in sent_array]
    print(word_array)

    dictionary = gensim.corpora.Dictionary(word_array)
    print(dictionary.token2id)

    # Create a corpus and pass the tokenized list of words to the Dictionary.doc2bow()
    # Here bow stands for bag-of-words
    corpus_source = [dictionary.doc2bow(word) for word in word_array]

    print(corpus_source)

    tfidf_source = gensim.models.TfidfModel(corpus_source)

    for doc in tfidf_source[corpus_source]:
        print([[dictionary[id], numpy.around(freq, decimals=2)] for id, freq in doc])

    return



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

    folder = ''
    
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
        print (str(err))

    # Loop continously checking the folder
    i = 1
    while True:

        filesList = [f for f in listdir(folder) if isfile(join(folder, f))]

        print(i, filesList)

        for fileName in filesList:
            fileType = check_filetype(folder+fileName)

            if fileType == "ASCII":
                print("File type: ASCII")
                vectorize_text(folder+fileName)
            elif fileType == "JPEG":
                print("File type: JPEG")
                vectorize_image(folder+fileName)
            else:
                print("Unknown file type, skipping...")

        i += 1

        if i > 1:
            sys.exit(2)

        time.sleep(10)


