import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["torontodb"]
myCollection = mydb["fileList"]


print(myclient.list_database_names())

fileRecord = { "fileType": "TEXT", "fileCrc": "0xfff", "fileSize": 0, "fileName": "Default File", "fileData": "empty" }
x = myCollection.insert_one(fileRecord)
print(x.inserted_id)
