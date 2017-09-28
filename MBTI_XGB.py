# -*- coding: utf-8 -*-

"""MBTI_XGB module

This module uses extreme boosted trees (xgboost) to determine personality
types from their posts.

Example:
         predicted_mbti = MBTI_XGB(list_of_posts)

"""

import pandas as pd
import os
import numpy as np
import pickle
import re
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def check_remove(FileName):
	'''
	This cehck if a file exists and remove it if yes.
	'''
	if os.path.isfile(FileName):
		os.remove(FileName)

def plot_confusion_matrix(cm, classes,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	"""
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))

	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def MBTI_XGB(X, y=[], save_txt=False):
	"""MBTI personality prediction with extrem grdaient boosting trees

	This function takes a list of posts and predict the mbti personality given these.
	If y specified, a score and confusion matrix is outputed.

	Args:
	    X (list): 		 List of posts froms different subject.
	    y (array):	 	 List of MBTI personalities.
	    save_txt (bool): If True, save a text file with predictions.

	Returns:
	    y_pred (array): The predicted mbti personalities.

	"""

	##### Encode each type to an int

	unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
						'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
	lab_encoder = LabelEncoder().fit(unique_type_list)
	if any(y):
		for i in y:
			if i not in unique_type_list:
				raise "not a MBTI personality"

	#### Compute list of subject with Type | Joint Comments

	# Lemmatize
	from nltk.corpus import stopwords
	from nltk.stem import WordNetLemmatizer

	# Lemmatize
	lemmatiser = WordNetLemmatizer()
	cachedStopWords = stopwords.words("english")

	def pre_process_data(X, y=[]):
		'''
		:return list of posts cleaned and list of personality labelized for xgboost
		'''

		print("Cleaning and processing data...")

		list_posts = []
		len_data = len(X)
		i = 0

		for posts in X:
			if type(posts) != str:
				raise "Not string find in X"
			i += 1
			if i % 500 == 0:
				print("%s | %s rows cleaned" % (i, len_data))

			##### Remove and clean comments
			temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
			temp = re.sub("[^a-zA-Z]", " ", temp)
			temp = re.sub(' +', ' ', temp).lower()
			temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])

			list_posts.append(temp)
		if any(y):
			list_personality = lab_encoder.transform(y)
		else:
			list_personality = []

		list_posts = np.array(list_posts)

		return list_posts, list_personality

	list_posts, list_personality = pre_process_data(X, y)

	# Load Vectorizer
	cntizer = pickle.load(open("data/CountVectorizer.pickle.dat", "rb"))

	print("Vectorizing words in posts...")
	train_X = cntizer.transform(list_posts)

	# Train with precalculated model
	xg_train = xgb.DMatrix(train_X)

	# Booster loaded
	bst = pickle.load(open("data/model.pickle.dat", "rb"))

	preds = bst.predict(xg_train)
	preds = np.array([np.argmax(prob) for prob in preds])

	if save_txt==True:
		print("\nSaving preds in data/MBTI_XGB_predictions.txt")

		check_remove('data/MBTI_XGB_predictions.txt')
		with open('data/MBTI_XGB_predictions.txt', 'a') as outfile:
			outfile.write("\n".join(lab_encoder.inverse_transform(preds.astype(int))))

	if any(y):
		score = classification_report(list_personality, preds)
		print('Classification report: \n%s' % score)
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(list_personality, preds)
		# Plot confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=lab_encoder.inverse_transform(range(16)),
							  title=('Confusion matrix normalized'))

# Example
'''
if __name__ == '__main__':

	dataset = 'path_to_dataset'

	data = pd.read_csv(dataset)

	X = data.posts.values
	y = data.type.values

	MBTI_XGB(X[1:200], y[1:200])
	MBTI_XGB(X, save_txt=True)
'''
