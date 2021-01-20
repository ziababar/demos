# import the required libraries
import numpy as np
import pymongo
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances


# Now, we are going to create similarity object using cosine similarity.
# Cosine similarity is a standard measure in Vector Space Modeling to determine the similarity of two vectors.
# The main class is Similarity, which builds an index for a given set of documents.


if __name__ == '__main__':

    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["torontodb"]
    myCollection = mydb["fileList"]

    myQuery = { "fileType" : "TEXT" }
    firstFilesList = list(myCollection.find(myQuery))
    secondFilesList = firstFilesList

    i = 0
    for firstFile in firstFilesList:

        j = 0

        while (j < len(firstFilesList)) and (i != j):

            secondFile = firstFilesList[j]

            x = np.asarray(firstFile.get("fileData", None))
            y = np.asarray(secondFile.get("fileData", None))
            cos_sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))
            if (cos_sim > 0.5):
                print ("Cosine Similarity between ", i, " and ", j, ": ", cos_sim)

            j = j + 1

        i = i + 1

