#!/usr/bin/env python

__author__ = 'metjush'

# This script handles web requests to learn a classification tree based on user-uploaded .csv dataset files
# Thus far, only a single Classification Tree is supported as the server needs to respond with a json
# and saving to json is only implemented for a single tree as of yet

# Receive the csv file
# Code based on http://code.activestate.com/recipes/273844-minimal-http-upload-cgi/
#           and http://webpython.codepoint.net/cgi_file_upload

import cgi, os, sys
import cgitb; cgitb.enable()
# hash the filename to create unreadable filename
import hashlib

filehash = hashlib.sha224()

try:
    import msvcrt
    msvcrt.setmode(0, os.O_BINARY)
    msvcrt.setmode(1, os.O_BINARY)
except ImportError:
    pass

UPLOAD_DIR = "datasets/"
JSON_DIR = "tmp_json/"

form = cgi.FieldStorage()

# get the file
fileitem = form['file']
depth = form['depth']
label_column = form['label']

# checking
if fileitem.filename:

    # strip leading path
    fn = os.path.basename(fileitem.filename)
    filehash.update(fn)
    name = filehash.hexdigest()
    open(UPLOAD_DIR + name + ".csv", 'wb').write(fileitem.file.read())
    savedfile = open(UPLOAD_DIR + name + ".csv", 'r')
else:
    raise IOError("Upload of file failed")

# we will be returning a json file, so set header
message_header = "header('Content-type: application/json');"

# Import ClassificationTree class
from ClassTree import ClassificationTree
import numpy as np




# read the saved file as a numpy array
data = np.loadtxt(savedfile, delimiter=",")
X = np.concatenate((data[:,0:label_column], data[:,(label_column+1):]))
y = data[:,label_column]

train_tree = ClassificationTree(depth_limit=depth)
train_tree.train()
train_json = train_tree.to_json(JSON_DIR + name + ".json")

