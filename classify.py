import numpy as np
from keras.preprocessing import image
import h5py
import sklearn
import tensorflow as tf
import os
import numpy as np

import matplotlib
import PIL.Image
import PIL.ImageSequence

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.svm import LinearSVC
from sklearn import svm, datasets, metrics 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,ShuffleSplit,RepeatedStratifiedKFold,cross_validate,KFold,cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE

import collections

import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.utils import to_categorical
import keras.backend as K


from imblearn.metrics import classification_report_imbalanced

import cv2
from scipy import ndimage

from utils import *
from feat_extract import * 

class classify(object):
	def __init__(self,
		path=None,
		weights=None,
		pooling='avg',
		cnn_layer=5,
		rotate=False,
		fuse=False,
		fusion=[4,5],
		fakes=False,
		fakes_path=None,
		rebalance=False,
		rebalance_size=100,
		number_of_fakes=None,
		iterations=1000,
		norm=True,
		pca=False,
		dim_red=None,
		retain_info=0.9,
		smote_fct=False,
		adasyn=False,
		k_neighbors=5,
		stratkfold=3,
		shuffle=True,
		repeated=None):

		self.path=path
		self.weights=weights
		self.pooling=pooling
		self.cnn_layer=cnn_layer
		self.rotate=rotate
		self.fuse=fuse
		self.fusion=fusion
		self.fakes=fakes
		self.fakes_path=fakes_path
		self.rebalance=rebalance
		self.rebalance_size=rebalance_size
		self.number_of_fakes=number_of_fakes
		self.iterations=iterations
		self.norm=norm
		self.pca=pca
		self.dim_red=dim_red
		self.retain_info=retain_info
		self.smote_fct=smote_fct
		self.adasyn=adasyn
		self.k_neighbors=k_neighbors
		self.stratkfold=stratkfold
		self.shuffle=shuffle
		self.repeated=repeated

		self.X = np.empty(shape=[0, 0])
		self.y = np.empty(shape=[0])

	def get_vectors(self):
		### Prepare Dataset
		### Extract features, get labels
		if self.fuse==True:
			self.X= feature_extract(path=self.path,weights=self.weights,pooling=self.pooling,layer=self.fusion).get_fusion()
			self.y= get_labels(self.path,self.X.shape[0])
		else:
			self.X = feature_extract(path=self.path,weights=self.weights,pooling=self.pooling,layer=self.cnn_layer).get_bottleneck()
			self.y = get_labels(self.path,self.X.shape[0])
			if rotate:
				path_aug = self.path+'rotated'
				X_aug = get_bottleneck(path=path_aug,weights=weights,pooling=pooling,layer=cnn_layer)
				y_aug = get_labels(path_aug,X.shape[0])
		### Add optional Fakes
		if self.fakes:
			X_fake = feature_extract(path=self.fakes_path,weights=self.weights,pooling=self.pooling, layer=self.cnn_layer).get_bottleneck()
			y_fake = get_labels(self.fakes_path,X_fake.shape[0])

			# Add certain ammount of fakes    
			X_fake_new = np.empty((0,X_fake.shape[1]))
			y_fake_new = np.array([])
			if self.number_of_fakes:
				for idx in range(0,len(Counter(y))):
					X_fake_new = np.append(X_fake_new,X_fake[int(idx*(X_fake.shape[0]/len(Counter(y)))):int(idx*(X_fake.shape[0]/len(Counter(y))))+number_of_fakes,:], axis=0)
					y_fake_new = np.append(y_fake_new,y_fake[int(idx*(X_fake.shape[0]/len(Counter(y)))):int(idx*(X_fake.shape[0]/len(Counter(y))))+number_of_fakes], axis=0)
				X_fake = X_fake_new
				y_fake = y_fake_new
			# Add to rebalance classes or just add all
			if self.rebalance:
				self.X, self.y = combine_data_rebalanced(self.rebalance_size,self.X,self.y,X_fake,y_fake)
			else:
				self.X, self.y = combine_data(self.X,self.y,X_fake,y_fake)


	def get_vectors2(self,Xtr_files,ytr,Xts_files,yts):
		fe_args = dict(
			path=self.path,
			weights=self.weights,
			pooling=self.pooling)
		X_fake =np.array([])
		y_fake =np.array([])
		Xtr_aug=np.array([])
		ytr_aug=np.array([])
		### Prepare Dataset
		### Extract features, get labels
		if self.fuse==True:
			Xtr= feature_extract(**fe_args,files=Xtr_files,layer=self.fusion).get_fusion()
			Xts= feature_extract(**fe_args,files=Xts_files,layer=self.fusion).get_fusion()
		else:
			Xtr = feature_extract(**fe_args,files=Xtr_files,layer=self.cnn_layer).get_bottleneck_files()
			Xts = feature_extract(**fe_args,files=Xts_files,layer=self.cnn_layer,rotate=False).get_bottleneck_files()
			if self.rotate:
				Xtr_aug = feature_extract(**fe_args,files=Xtr_files,rotate=self.rotate).get_bottleneck_aug_files2()
				ytr_aug = get_labels_aug(ytr,Xtr_aug.shape[0])

			
		### Add optional Fakes
		if self.fakes:
			X_fake = feature_extract(path=self.fakes_path,weights=self.weights,pooling=self.pooling, layer=self.cnn_layer).get_bottleneck()
			y_fake = get_labels(self.fakes_path,X_fake.shape[0])
			print(X_fake.shape)
			# Add certain ammount of fakes    
			X_fake_new = np.empty((0,X_fake.shape[1]))
			y_fake_new = np.array([])
			if self.number_of_fakes:
				for idx in range(0,len(Counter(ytr))):
					X_fake_new = np.append(X_fake_new,X_fake[idx*(X_fake.shape[0]//len(Counter(ytr))):idx*(X_fake.shape[0]//len(Counter(ytr)))+self.number_of_fakes,:], axis=0)
					y_fake_new = np.append(y_fake_new,y_fake[idx*(X_fake.shape[0]//len(Counter(ytr))):idx*(X_fake.shape[0]//len(Counter(ytr)))+self.number_of_fakes], axis=0)
				X_fake = X_fake_new
				y_fake = y_fake_new
				print(X_fake.shape)
			# Add to rebalance classes or just add all
			#if self.rebalance:
			#	self.X, self.y = combine_data_rebalanced(self.rebalance_size,self.X,self.y,X_fake,y_fake)
			#else:
			#	self.X, self.y = combine_data(self.X,self.y,X_fake,y_fake)
		return Xtr,ytr,Xts,yts,Xtr_aug,ytr_aug,X_fake,y_fake

	def process(self):
		'''
		Process vectors once obtained
		normalize
		pca/ctv
		optional: smote/adaysn
		'''
		### Normalize & PCA the features
		if self.norm==True:
			self.X = StandardScaler().fit(self.X).transform(self.X)
		if self.pca==True:
			if self.dim_red:
				pca = PCA(dim_red).fit(self.X)
			else:
				pca = PCA().fit(self.X)
				if self.retain_info==1:
					info_retain = np.argmax(pca.explained_variance_ratio_.cumsum())
					pca = PCA(info_retain).fit(self.X)
				else:
					info_retain = np.where(pca.explained_variance_ratio_.cumsum() >= self.retain_info)
					pca = PCA(info_retain[0][0]).fit(self.X)
			self.X = pca.transform(self.X)

		### Optional Oversampling SMOTE or ADAYSN
		if self.smote_fct==True:
			print(Counter(y))
			sm = SMOTE(sampling_strategy='not majority',random_state=41, k_neighbors=self.k_neighbors)


			self.X, self.y = sm.fit_resample(self.X[1:,:], self.y[1:])
			print('Resampled dataset shape %s' % Counter(self.y))

		if self.adasyn==True:
			print('Original dataset shape %s' % Counter(self.y))
			ada = ADASYN(random_state=42,n_neighbors=self.k_neighbors)
			self.X, self.y = ada.fit_resample(self.X, self.y)
			print('Resampled dataset shape %s' % Counter(self.y))

	def process2(self,Xtr,ytr,Xts):
		# print(Xtr.shape,Xts.shape)
		# x1 = Xtr.shape[0]
		# x2 = Xts.shape[0]
		# X = np.concatenate(Xtr,Xts,axis=0)
		# if self.norm==True:
		# 	X = StandardScaler().fit(X).transform(X)
		# if self.pca==True:
		# 	if self.dim_red:
		# 		pca = PCA(dim_red).fit(X)
		# 	else:
		# 		pca = PCA().fit(X)
		# 		if self.retain_info==1:
		# 			info_retain = np.argmax(pca.explained_variance_ratio_.cumsum())
		# 			pca = PCA(info_retain).fit(X)
		# 		else:
		# 			info_retain = np.where(pca.explained_variance_ratio_.cumsum() >= self.retain_info)
		# 			if (info_retain[0][0]>0):
		# 				pca = PCA(info_retain[0][0]).fit(X)
		# 			elif (info_retain[0][0]==0): #Randbedingung, s.d. wir nicht auf 0 dim reduzieren
		# 				pca = PCA(1).fit(X)
		# 	X = pca.transform(X)
		# 	Xtr = X[:x1,:]
		# 	Xts = X[x1:,:]
		# print(Xtr.shape,Xts.shape)
		if self.pca==True:
			if self.dim_red:
				pca = PCA(dim_red).fit(Xtr)
				#pca = PCA(dim_red).fit(Xts)
			else:
				pca = PCA().fit(Xtr)
				#pca = PCA().fit(Xts)
				if self.retain_info==1:
					info_retain = np.argmax(pca.explained_variance_ratio_.cumsum())
					pca = PCA(info_retain).fit(Xtr)
					#pca = PCA(info_retain).fit(Xts)
				else:
					info_retain = np.where(pca.explained_variance_ratio_.cumsum() >= self.retain_info)
					if (info_retain[0][0]>0):
						pca = PCA(info_retain[0][0]).fit(Xtr)
						#pca = PCA(info_retain[0][0]).fit(Xts)
					elif (info_retain[0][0]==0): #Randbedingung, s.d. wir nicht auf 0 dim reduzieren
						pca = PCA(1).fit(Xtr)
						#pca = PCA(1).fit(Xts)

			Xtr = pca.transform(Xtr)
			Xts = pca.transform(Xts)
		
		### Optional Oversampling SMOTE or ADAYSN
		if self.smote_fct==True:
			sm = SVMSMOTE(sampling_strategy='not majority',random_state=41, k_neighbors=self.k_neighbors)
			Xtr, ytr = sm.fit_resample(Xtr, ytr)
			print('Resampled dataset shape %s' % Counter(ytr))

		if self.adasyn==True:
			print('Original dataset shape %s' % Counter(ytr))
			ada = ADASYN(random_state=42,n_neighbors=self.k_neighbors)
			Xtr, ytr = ada.fit_resample(Xtr, ytr)
			print('Resampled dataset shape %s' % Counter(ytr))
		return Xtr,ytr,Xts

	def svm_bottleneck(self):

		
		X_files = [os.path.join(self.path,fold,x) for fold in sorted(os.listdir(self.path)) for x in os.listdir(os.path.join(self.path,fold))]
		y = get_labels(self.path,len(X_files))

		class_names = [name for name in sorted(os.listdir(self.path))]
		print(class_names)
		## Init SVM
		clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr',max_iter=self.iterations)

		# Validation method
		#kFold = StratifiedKFold(n_splits=self.stratkfold, shuffle=self.shuffle)
		kFold = RepeatedStratifiedKFold(n_splits=self.stratkfold,n_repeats=5)
		#kFold = StratifiedShuffleSplit(n_splits=10,test_size=0.2)

		c = []
		p = []
		r = []
		f = []
		c_x = [[[],[],[],[],[],[]] for i in enumerate(os.listdir(self.path))]
		for idx,class_name in enumerate(os.listdir(self.path)):
			c_x[idx] = [class_name,[],[],[],[],[]]

		i=1
		pesudo_few = False
		zero_shot = False
		for train_idx, test_idx in kFold.split(X_files,y):
			X_train = [X_files[idx] for idx in train_idx]
			X_test = [X_files[idx] for idx in test_idx]
			y_train = [y[idx] for idx in train_idx]
			y_test = [y[idx] for idx in test_idx]

			Xtr,ytr,Xts,yts,Xtr_aug,ytr_aug,Xtr_fake,ytr_fake = self.get_vectors2(X_train,y_train,X_test,y_test)
			print("Base:", Xtr.shape, "Aug:", Xtr_aug.shape, "Fake:", Xtr_fake.shape)
			print("Train:", Xtr.shape[0], Counter(ytr))
			print("Test:", Xts.shape[0], Counter(yts))
			print("Aug:", Xtr_aug.shape[0], Counter(ytr_aug))
			print("Fake:", Xtr_fake.shape[0], Counter(ytr_fake))
			if pesudo_few:
				if self.rotate and self.fakes:
					tmp_x = np.concatenate((Xtr_aug,Xtr_fake),axis=0)
					tmp_y = np.concatenate((ytr_aug,ytr_fake),axis=0)
				elif self.rotate:
					tmp_x = Xtr_aug
					tmp_y = ytr_aug
				elif self.fakes:
					tmp_x = Xtr_fake
					tmp_y = ytr_fake
				Xts = np.concatenate((Xts,Xtr),axis=0)
				yts = np.concatenate((yts,ytr),axis=0)
				Xtr = tmp_x
				ytr = tmp_y
			else:
				if self.rotate:		
					Xtr = np.concatenate((Xtr,Xtr_aug),axis=0)
					ytr = np.concatenate((ytr,ytr_aug),axis=0)
				if self.fakes:
					Xtr = np.concatenate((Xtr,Xtr_fake),axis=0)
					ytr = np.concatenate((ytr,ytr_fake),axis=0)			
			if zero_shot:
				print(np.where(ytr==6))
			Xtr,ytr,Xts = self.process2(Xtr,ytr,Xts)

			print("Training samples:", Counter(ytr), Xtr.shape)
			print("Test samples:", Counter(yts), Xts.shape)

			clf.fit(Xtr, ytr)
			y_pred = clf.predict(Xts)

			accuracy = accuracy_score(yts, y_pred)*100
			precision_s = precision_score(yts,y_pred,average='weighted')*100
			recall_s = recall_score(yts,y_pred,average='weighted')*100
			f1_s = f1_score(yts,y_pred,average='weighted')*100

			c_i = [[[],[],[],[],[],[]] for i in enumerate(os.listdir(self.path))]
			for idx, class_fold  in enumerate(os.listdir(self.path)):
				c_i[idx] = single_class_report([yts[x] for x in np.where(np.array(yts)==idx)[0]],[y_pred[x] for x in np.where(np.array(yts)==idx)[0]]) 

			#print(classification_report_imbalanced(yts, y_pred, target_names=class_names))

			c.append(accuracy)
			p.append(precision_s)
			r.append(recall_s)
			f.append(f1_s)

			for idx,_ in enumerate(os.listdir(self.path)):
				for m in range(0,len(c_i[0])):
					c_x[idx][m+1].append(c_i[idx][m])
			i+=1
		#plot_confusion_matrix(y_test, y_pred, classes=[name for name in sorted(os.listdir(path))], normalize=True,title='Pleophylla Pre-Trained')
		return [Counter(self.y),
				[np.mean(c),np.std(c), np.mean(p),np.mean(r),np.mean(f)],
				[[c_x[idx][0],np.mean(c_x[idx][1]),np.std(c_x[idx][1]),np.mean(c_x[idx][2]),np.mean(c_x[idx][3]), np.mean(c_x[idx][4]),np.mean(c_x[idx][5])] for idx,_ in enumerate(os.listdir(self.path))]]	

	def svm_all_features(self):
		### Prepare Dataset
		### Extract features, get labels
		dictionary = feature_extract(path=self.path,weights=self.weights,pooling=self.pooling,layer=self.cnn_layer).get_features()
		tmp=[]
		for x_tmp in dictionary:
			self.X = x_tmp
			self.y = get_labels(path,X.shape[0])

			self.process()
			## Init SVM
			clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr',max_iter=self.iterations)
			# Validation method
			kFold = StratifiedKFold(n_splits=self.stratkfold, shuffle=self.shuffle)
			#kFold = RepeatedStratifiedKFold(n_splits=stratkfold,n_repeats=5)
			#kFold = StratifiedShuffleSplit(n_splits=10,test_size=0.2)

			c = []
			p = []
			r = []
			f = []
			c_x = [[[],[],[],[],[],[]] for i in enumerate(os.listdir(self.path))]
			for idx,class_name in enumerate(os.listdir(self.path)):
				c_x[idx] = [class_name,[],[],[],[],[]]

			i=1
			print(self.X.shape)
			for train_idx, test_idx in kFold.split(self.X,self.y):
				X_train = [self.X[idx] for idx in train_idx]
				X_test = [self.X[idx] for idx in test_idx]
				y_train = [self.y[idx] for idx in train_idx]
				y_test = [self.y[idx] for idx in test_idx]


				clf.fit(X_train, y_train)
				y_pred = clf.predict(X_test)

				accuracy = accuracy_score(y_test, y_pred)*100
				precision_s = precision_score(y_test,y_pred,average='weighted')*100
				recall_s = recall_score(y_test,y_pred,average='weighted')*100
				f1_s = f1_score(y_test,y_pred,average='weighted')*100

				c_i = [[[],[],[],[],[],[]] for i in enumerate(os.listdir(self.path))]
				for idx, class_fold  in enumerate(os.listdir(self.path)):
					c_i[idx] = single_class_report([y_test[x] for x in np.where(np.array(y_test)==idx)[0]],[y_pred[x] for x in np.where(np.array(y_test)==idx)[0]]) 

				#print(classification_report_imbalanced(y_test, y_pred, target_names=[name for name in sorted(os.listdir(path))]))

				c.append(accuracy)
				p.append(precision_s)
				r.append(recall_s)
				f.append(f1_s)

				for idx,_ in enumerate(os.listdir(self.path)):
					for m in range(0,len(c_i[0])):
						c_x[idx][m+1].append(c_i[idx][m])
				i+=1
			#plot_confusion_matrix(y_test, y_pred, classes=[name for name in sorted(os.listdir(path))], normalize=True,title='Pleophylla Pre-Trained')
			x = [Counter(self.y),
					[np.mean(c),np.std(c), np.mean(p),np.mean(r),np.mean(f)],
					[[c_x[idx][0],np.mean(c_x[idx][1]),np.std(c_x[idx][1]),np.mean(c_x[idx][2]),np.mean(c_x[idx][3]), np.mean(c_x[idx][4]),np.mean(c_x[idx][5])] for idx,_ in enumerate(os.listdir(self.path))]
					]
			tmp.append(x)
		return tmp, self.X,self.y	