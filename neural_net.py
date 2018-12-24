import text_preprocessor
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from text_preprocessor import *

word_index = create_dictionary()
csv_loc = "C:/Users/Stephen/Desktop/4444-Group-Project/neural_net/question-set.csv"
df = pd.read_csv(csv_loc, sep = ",", header = 0)

email_test_data = []
email_train_data = []
email_test_labels = []
email_train_labels = []
"""
cc_test_data = []
cc_train_data = []
ssn_test_data = 
ssn_train_data = []
ads_test_data = 
ads_train_data = []
location_test_data = 
location_train_data = []
pii_test_data = 
pii_train_data = []
law_test_data = 
law_train_data = []
change_test_data = 
change_train_data = []
control_test_data = 
control_train_data = []
agg_test_data = 
agg_train_data = []

cc_test_labels = 
cc_train_labels = []
ssn_test_labels = 
ssn_train_labels = []
ads_test_labels = 
ads_train_labels = []
location_test_labels = 
location_train_labels = []
pii_test_labels = 
pii_train_labels = []
law_test_labels = 
law_train_labels = []
change_test_labels = 
change_train_labels = []
control_test_labels = 
control_train_labels = []
agg_test_labels = 
agg_train_labels = []
"""

for i in range(2, 400):
	outputString = ""
	try:
		outputString = " ".join(create_output("UTA/%i.txt" % i)[0])
		email_train_data.append(outputString.split(" "))
		email_train_labels.append(int(df["Email"][i-2]) % 3)
	except:
		continue

for i in range(len(email_train_data)):
	for j in range(len(email_train_data[i])):
		email_train_data[i][j] = word_index[email_train_data[i][j]]

for i in range(400, 456):
	outputString = ""
	try:
		outputString = " ".join(create_output("UTA/%i.txt" % i)[0])
		email_test_data.append(outputString.split(" "))
		email_test_labels.append(int(df["Email"][i-2]) % 3)
	except:
		continue

for i in range(len(email_test_data)):
	for j in range(len(email_test_data[i])):
		email_test_data[i][j] = word_index[email_test_data[i][j]]


with tf.device("/gpu:0"):
	email_train_data = keras.preprocessing.sequence.pad_sequences(email_train_data,
	                                                        value = word_index["<PAD>"],
	                                                        padding = "post",
	                                                        maxlen = 256)

	email_test_data = keras.preprocessing.sequence.pad_sequences(email_test_data,
	                                                       value = word_index["<PAD>"],
	                                                       padding = "post",
	                                                       maxlen = 256)

	vocab_size = 10854

	model = keras.Sequential()
	model.add(keras.layers.Embedding(vocab_size, 16))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(16, activation = tf.nn.relu))
	model.add(keras.layers.Dense(3, activation = tf.nn.sigmoid))

	model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

	# model.fit(email_train_data, email_train_labels, epochs = 5)

	x_val = email_train_data[:100]
	partial_x_train = email_train_data[100:]

	y_val = email_train_labels[:100]
	partial_y_train = email_train_labels[100:]

	history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 40,
                    batch_size = 50,
                    validation_data = (x_val, y_val),
                    verbose = 1)

	results = model.evaluate(email_train_data, email_train_labels)

	print(results)

	history_dict = history.history
	history_dict.keys()
	dict_keys(["val_acc", "acc", "val_loss", "loss"])

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(acc) + 1)

	# "bo" is for "blue dot"
	plt.plot(epochs, loss, 'bo', label='Training loss')
	# b is for "solid blue line"
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.show()

	plt.clf()  # clear figure
	acc_values = history_dict['acc']
	val_acc_values = history_dict['val_acc']

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.show()

