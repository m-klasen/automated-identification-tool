from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import keras.backend as K
import PIL.Image
import PIL.ImageSequence
from utils import *
from scipy import ndimage
import cv2

class feature_extract(object):	
	def __init__(self,
		path=None,
		files=None,
		weights=None,
		pooling='avg',
		rotate=False,
		input_shape=(224,224,3),
		target_shape=(224,224), 
		layer=None):

		self.path = path
		self.files = files
		self.weights = weights
		self.pooling = pooling
		self.rotate = rotate
		self.input_shape = input_shape
		self.target_shape = target_shape
		self.layer = layer

	def get_features2(self):
		w_string = self.weights.split('/')[1].split('.')[0]
		if os.path.exists('features/'+self.path+"/block1_conv2_"+self.pooling+"_features_"+w_string+".npy") and not (self.weights is None):
			featb1c2_= np.load('features/'+self.path+"/block1_conv2_"+self.pooling+"_features_"+w_string+".npy")
			featb2c2_= np.load('features/'+self.path+"/block2_conv2_"+self.pooling+"_features_"+w_string+".npy")
			featb3c3_= np.load('features/'+self.path+"/block3_conv3_"+self.pooling+"_features_"+w_string+".npy")
			featb4c3_= np.load('features/'+self.path+"/block4_conv3_"+self.pooling+"_features_"+w_string+".npy")
			featb5c3_= np.load('features/'+self.path+"/block5_conv3_"+self.pooling+"_features_"+w_string+".npy")
			print("Features loaded from File")  
		else:
			if self.path==None:
				raise Exception('no file path')
			if self.weights==None or self.weights=='pretrained':
				model = applications.VGG16(weights='imagenet', include_top=False,input_shape=self.input_shape)
				print('Imagenet loaded')
			else:
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Weights Loaded')
			modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block5_conv3').output)
			modelb4c3 = Model(inputs=model.input, outputs=model.get_layer('block4_conv3').output)
			modelb3c3 = Model(inputs=model.input, outputs=model.get_layer('block3_conv3').output)
			modelb2c2 = Model(inputs=model.input, outputs=model.get_layer('block2_conv2').output)
			modelb1c2 = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)
			print('Model loaded.')
			# Create new Model to add global pooling layer
			modelb5c3end = Sequential()
			modelb4c3end = Sequential()
			modelb3c3end = Sequential()
			modelb2c2end = Sequential()
			modelb1c2end = Sequential()
			if pooling=='max':
				#Block5
				modelb5c3end.add(GlobalMaxPooling2D())
				#Block4			
				modelb4c3end.add(GlobalMaxPooling2D())			
				#Block3			
				modelb3c3end.add(GlobalMaxPooling2D())			
				#BLock2			
				modelb2c2end.add(GlobalMaxPooling2D())			
				#Block1			
				modelb1c2end.add(GlobalMaxPooling2D())			
			elif pooling=='avg':
				#Block5
				modelb5c3end.add(GlobalAveragePooling2D())
				#Block4
				modelb4c3end.add(GlobalAveragePooling2D())
				#Block3
				modelb3c3end.add(GlobalAveragePooling2D())
				#BLock2
				modelb2c2end.add(GlobalAveragePooling2D())
				#Block1
				modelb1c2end.add(GlobalAveragePooling2D())
			else:
				raise Exception('invalid pooling, avg or max')
			modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
			modelb4c3end = Model(inputs=modelb4c3.input, outputs=modelb4c3end(modelb4c3.output))
			modelb3c3end = Model(inputs=modelb3c3.input, outputs=modelb3c3end(modelb3c3.output))
			modelb2c2end = Model(inputs=modelb2c2.input, outputs=modelb2c2end(modelb2c2.output))
			modelb1c2end = Model(inputs=modelb1c2.input, outputs=modelb1c2end(modelb1c2.output))
			z = []
			z_val = []
			i=0
			j=0
			for file in path:
				img = image.load_img(file, target_size=self.target_shape)
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)

				featb1c2 = modelb1c2end.predict(x)
				featb1c2 = featb1c2.reshape((1,featb1c2.shape[1]))

				featb2c2 = modelb2c2end.predict(x)
				featb2c2 = featb2c2.reshape((1,featb2c2.shape[1]))

				featb3c3 = modelb3c3end.predict(x)
				featb3c3 = featb3c3.reshape((1,featb3c3.shape[1]))

				featb4c3 = modelb4c3end.predict(x)
				featb4c3 = featb4c3.reshape((1,featb4c3.shape[1]))

				featb5c3 = modelb5c3end.predict(x)
				featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]))
				if i==0:
					featb1c2_ = featb1c2
					featb2c2_ = featb2c2
					featb3c3_ = featb3c3
					featb4c3_ = featb4c3
					featb5c3_ = featb5c3
					i+=1
				else:    
					featb1c2_ = np.append(featb1c2_, featb1c2, axis=0)
					featb2c2_ = np.append(featb2c2_, featb2c2, axis=0)
					featb3c3_ = np.append(featb3c3_, featb3c3, axis=0)
					featb4c3_ = np.append(featb4c3_, featb4c3, axis=0)
					featb5c3_ = np.append(featb5c3_, featb5c3, axis=0)
				j+=1

			print(str(j) + " features loaded")
			os.makedirs('features/'+self.path,exist_ok=True)
			if not (self.weights is None):
				np.save('features/'+self.path+"/block1_conv2_"+self.pooling+"_features_"+w_string+".npy",featb1c2_)
				np.save('features/'+self.path+"/block2_conv2_"+self.pooling+"_features_"+w_string+".npy",featb2c2_)
				np.save('features/'+self.path+"/block3_conv3_"+self.pooling+"_features_"+w_string+".npy",featb3c3_)
				np.save('features/'+self.path+"/block4_conv3_"+self.pooling+"_features_"+w_string+".npy",featb4c3_)
				np.save('features/'+self.path+"/block5_conv3_"+self.pooling+"_features_"+w_string+".npy",featb5c3_)
		dict_ = [featb1c2_,featb2c2_,featb3c3_,featb4c3_,featb5c3_]
		return dict_

	def get_features(self):
		w_string = self.weights.split('/')[1].split('.')[0]
		if os.path.exists('features/'+self.path+"/block1_conv2_"+self.pooling+"_features_"+w_string+".npy") and not (self.weights is None):
			featb1c2_= np.load('features/'+self.path+"/block1_conv2_"+self.pooling+"_features_"+w_string+".npy")
			featb2c2_= np.load('features/'+self.path+"/block2_conv2_"+self.pooling+"_features_"+w_string+".npy")
			featb3c3_= np.load('features/'+self.path+"/block3_conv3_"+self.pooling+"_features_"+w_string+".npy")
			featb4c3_= np.load('features/'+self.path+"/block4_conv3_"+self.pooling+"_features_"+w_string+".npy")
			featb5c3_= np.load('features/'+self.path+"/block5_conv3_"+self.pooling+"_features_"+w_string+".npy")
			print("Features loaded from File")  
		else:
			if self.path==None:
				raise Exception('no file path')
			if self.weights==None or self.weights=='pretrained':
				model = applications.VGG16(weights='imagenet', include_top=False,input_shape=self.input_shape)
				print('Imagenet loaded')
			else:
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Weights Loaded')
			modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block5_conv3').output)
			modelb4c3 = Model(inputs=model.input, outputs=model.get_layer('block4_conv3').output)
			modelb3c3 = Model(inputs=model.input, outputs=model.get_layer('block3_conv3').output)
			modelb2c2 = Model(inputs=model.input, outputs=model.get_layer('block2_conv2').output)
			modelb1c2 = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)
			print('Model loaded.')
			# Create new Model to add global pooling layer
			modelb5c3end = Sequential()
			modelb4c3end = Sequential()
			modelb3c3end = Sequential()
			modelb2c2end = Sequential()
			modelb1c2end = Sequential()
			if pooling=='max':
				#Block5
				modelb5c3end.add(GlobalMaxPooling2D())
				#Block4			
				modelb4c3end.add(GlobalMaxPooling2D())			
				#Block3			
				modelb3c3end.add(GlobalMaxPooling2D())			
				#BLock2			
				modelb2c2end.add(GlobalMaxPooling2D())			
				#Block1			
				modelb1c2end.add(GlobalMaxPooling2D())			
			elif pooling=='avg':
				#Block5
				modelb5c3end.add(GlobalAveragePooling2D())
				#Block4
				modelb4c3end.add(GlobalAveragePooling2D())
				#Block3
				modelb3c3end.add(GlobalAveragePooling2D())
				#BLock2
				modelb2c2end.add(GlobalAveragePooling2D())
				#Block1
				modelb1c2end.add(GlobalAveragePooling2D())
			else:
				raise Exception('invalid pooling, avg or max')
			modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
			modelb4c3end = Model(inputs=modelb4c3.input, outputs=modelb4c3end(modelb4c3.output))
			modelb3c3end = Model(inputs=modelb3c3.input, outputs=modelb3c3end(modelb3c3.output))
			modelb2c2end = Model(inputs=modelb2c2.input, outputs=modelb2c2end(modelb2c2.output))
			modelb1c2end = Model(inputs=modelb1c2.input, outputs=modelb1c2end(modelb1c2.output))
			z = []
			z_val = []
			i=0
			j=0
			for fold in sorted(os.listdir(self.path)):
				for file in os.listdir(self.path+"/"+fold):
					file_to_path=self.path+"/"+fold+"/"+file
					img = image.load_img(file_to_path, target_size=self.target_shape)
					x = image.img_to_array(img)
					x = np.expand_dims(x, axis=0)


					featb1c2 = modelb1c2end.predict(x)
					featb1c2 = featb1c2.reshape((1,featb1c2.shape[1]))

					featb2c2 = modelb2c2end.predict(x)
					featb2c2 = featb2c2.reshape((1,featb2c2.shape[1]))

					featb3c3 = modelb3c3end.predict(x)
					featb3c3 = featb3c3.reshape((1,featb3c3.shape[1]))

					featb4c3 = modelb4c3end.predict(x)
					featb4c3 = featb4c3.reshape((1,featb4c3.shape[1]))

					featb5c3 = modelb5c3end.predict(x)
					featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]))
					if i==0:
						featb1c2_ = featb1c2
						featb2c2_ = featb2c2
						featb3c3_ = featb3c3
						featb4c3_ = featb4c3
						featb5c3_ = featb5c3
						i+=1
					else:    
						featb1c2_ = np.append(featb1c2_, featb1c2, axis=0)
						featb2c2_ = np.append(featb2c2_, featb2c2, axis=0)
						featb3c3_ = np.append(featb3c3_, featb3c3, axis=0)
						featb4c3_ = np.append(featb4c3_, featb4c3, axis=0)
						featb5c3_ = np.append(featb5c3_, featb5c3, axis=0)
					j+=1

			print(str(j) + " features loaded")
			os.makedirs('features/'+self.path,exist_ok=True)
			if not (self.weights is None):
				np.save('features/'+self.path+"/block1_conv2_"+self.pooling+"_features_"+w_string+".npy",featb1c2_)
				np.save('features/'+self.path+"/block2_conv2_"+self.pooling+"_features_"+w_string+".npy",featb2c2_)
				np.save('features/'+self.path+"/block3_conv3_"+self.pooling+"_features_"+w_string+".npy",featb3c3_)
				np.save('features/'+self.path+"/block4_conv3_"+self.pooling+"_features_"+w_string+".npy",featb4c3_)
				np.save('features/'+self.path+"/block5_conv3_"+self.pooling+"_features_"+w_string+".npy",featb5c3_)
		dict_ = [featb1c2_,featb2c2_,featb3c3_,featb4c3_,featb5c3_]
		return dict_

	def get_bottleneck(self):
		print(self.weights)
		w_string = self.weights.split('/')[1].split('.')[0]
		feature_path = "features/"+self.path+"/block"+str(self.layer)+"_conv3_"+self.pooling+"_features_"+w_string+".npy"
		if(os.path.exists(feature_path) and self.weights!='pretrained'):
			featb5c3_= np.load(feature_path)
			print("Features loaded from File") 
		else:
			if self.path==None:
				raise Exception('no file path')
			if self.weights==None or self.weights=='weights/pretrained':
				model = applications.VGG16(weights='imagenet', include_top=False,input_shape=self.input_shape)
				print('Imagenet loaded')
			else:
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Weights Loaded')
			modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block'+str(self.layer)+'_conv3').output)
			print('Model loaded.')
			if self.pooling=='max':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalMaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Max Pooling')
			elif self.pooling=='avg':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalAveragePooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Avg Pooling')
			elif self.pooling=='normal':
			#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(MaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Max Pooling')
			z = []
			z_val = []
			i=0
			j=0
			for fold in sorted(os.listdir(self.path)):
				for file in os.listdir(self.path+"/"+fold):
					file_to_path=self.path+"/"+fold+"/"+file
					img = image.load_img(file_to_path, target_size=self.target_shape)
					x = image.img_to_array(img)
					x = np.expand_dims(x, axis=0)

					featb5c3 = modelb5c3end.predict(x)
					try:
						featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]))
					except:
						featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]*featb5c3.shape[2]*featb5c3.shape[3]))
					if i==0:
						featb5c3_ = featb5c3
						i+=1
					else:
						featb5c3_ = np.append(featb5c3_, featb5c3, axis=0)
					j+=1

			print(str(j) + " features loaded")
			os.makedirs('features/'+self.path,exist_ok=True)
			if (self.weights!='pretrained'):
				np.save(feature_path, featb5c3_ )
		return featb5c3_

	def get_bottleneck_files(self):
		w_string = self.weights.split('/')[1].split('.')[0]
		try:
			for file in self.files:
				feature_file_path = "features/"+file[9:-4]+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				file_feature = np.load(feature_file_path)
				try:
					featb5c3_ = np.append(featb5c3_,file_feature, axis=0)
				except:
					featb5c3_ = file_feature
		except:
			pass
			if self.path==None:
				raise Exception('no file path')
			if self.weights==None or self.weights=='weights/pretrained.h5':
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Imagenet loaded')
			else:
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Weights Loaded')
			modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block'+str(self.layer)+'_conv3').output)
			print('Model loaded.')
			if self.pooling=='max':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalMaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Max Pooling')
			elif self.pooling=='avg':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalAveragePooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Avg Pooling')
			elif self.pooling=='normal':
			#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(MaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Max Pooling')

			i=0
			j=0
			for file in self.files:
							
				img = image.load_img(file, target_size=self.target_shape)
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)	
				featb5c3 = modelb5c3end.predict(x)
				try:
					featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]))
				except:
					featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]*featb5c3.shape[2]*featb5c3.shape[3]))
				if i==0:
					featb5c3_ = featb5c3
					i+=1
				else:
					featb5c3_ = np.append(featb5c3_, featb5c3, axis=0)
				j+=1
				s = file.split('/')
				os.makedirs("features/"+s[1]+"/"+s[2],exist_ok=True)
				feature_file_path = "features/"+file[9:-4]+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				np.save(feature_file_path, featb5c3 )
			print(str(j) + " features loaded")
			print(featb5c3_.shape)
		return featb5c3_

	def get_bottleneck_aug_files2(self):
		w_string = self.weights.split('/')[1].split('.')[0]
		try:
			for file in self.files:
				for degree in range(5,25,5):
					feature_file_path = "features/augment2/"+file[9:-4]+"+"+str(degree)+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
					file_feature = np.load(feature_file_path)
					feature_file_path = "features/augment2/"+file[9:-4]+"-"+str(degree)+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
					file_feature2 = np.load(feature_file_path)
					try:
						featb5c3_ = np.append(featb5c3_,file_feature, axis=0)
						featb5c3_ = np.append(featb5c3_,file_feature2, axis=0)
					except:
						featb5c3_ = file_feature
						featb5c3_ = np.append(featb5c3_,file_feature2, axis=0)
		except:
			pass
			if self.path==None:
				raise Exception('no file path')
			if self.weights==None or self.weights=='weights/pretrained':
				model = applications.VGG16(weights='imagenet', include_top=False,input_shape=self.input_shape)
				print('Imagenet loaded')
			else:
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Weights Loaded')
			modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block'+str(self.layer)+'_conv3').output)
			print('Model loaded.')
			if self.pooling=='max':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalMaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Max Pooling')
			elif self.pooling=='avg':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalAveragePooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Avg Pooling')
			elif self.pooling=='normal':
			#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(MaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Max Pooling')

			i=0
			j=0
			for file in self.files:
				for degree in range(5,25,5):
					x = np.array(PIL.Image.open(file))
					img_rot1 = ndimage.rotate(x, -degree, reshape=True, mode='nearest')
					rot1 = np.expand_dims(cv2.resize(img_rot1,self.target_shape), axis=0)
					img_rot2 = ndimage.rotate(x, degree, reshape=True, mode='nearest')
					rot2 = np.expand_dims(cv2.resize(img_rot2,self.target_shape), axis=0)
					featb5c3 = modelb5c3end.predict(rot1)
					featb5c3_2 = modelb5c3end.predict(rot2)

					try:
						featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]))
					except:
						featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]*featb5c3.shape[2]*featb5c3.shape[3]))
					if i==0:
						featb5c3_ = featb5c3
						featb5c3_ = np.append(featb5c3_, featb5c3_2, axis=0)
						i+=1
					else:
						featb5c3_ = np.append(featb5c3_, featb5c3, axis=0)
						featb5c3_ = np.append(featb5c3_, featb5c3_2, axis=0)
					j+=1        
					s = file.split('/')
					os.makedirs("features/augment2/"+s[1]+"/"+s[2],exist_ok=True)
					feature_file_path = "features/augment2/"+file[9:-4]+"+"+str(degree)+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
					np.save(feature_file_path, featb5c3 )
					feature_file_path = "features/augment2/"+file[9:-4]+"-"+str(degree)+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
					np.save(feature_file_path, featb5c3_2 )
			print(str(j) + " features loaded")
		return featb5c3_
	def get_bottleneck_aug_files(self):
		w_string = self.weights.split('/')[1].split('.')[0]
		try:
			for file in self.files:
				feature_file_path = "features/augment/"+file[9:-4]+"+20"+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				file_feature = np.load(feature_file_path)
				feature_file_path = "features/augment/"+file[9:-4]+"-20"+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				file_feature2 = np.load(feature_file_path)
				try:
					featb5c3_ = np.append(featb5c3_,file_feature, axis=0)
					featb5c3_ = np.append(featb5c3_,file_feature2, axis=0)
				except:
					featb5c3_ = file_feature
					featb5c3_ = np.append(featb5c3_,file_feature2, axis=0)
		except:
			pass
			if self.path==None:
				raise Exception('no file path')
			if self.weights==None or self.weights=='weights/pretrained':
				model = applications.VGG16(weights='imagenet', include_top=False,input_shape=self.input_shape)
				print('Imagenet loaded')
			else:
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Weights Loaded')
			modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block'+str(self.layer)+'_conv3').output)
			print('Model loaded.')
			if self.pooling=='max':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalMaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Max Pooling')
			elif self.pooling=='avg':
				#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(GlobalAveragePooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Global Avg Pooling')
			elif self.pooling=='normal':
			#Block5
				modelb5c3end = Sequential()
				modelb5c3end.add(MaxPooling2D())
				modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
				print('Max Pooling')

			i=0
			j=0
			for file in self.files:
				x = np.array(PIL.Image.open(file))
				degree=20
				img_rot1 = ndimage.rotate(x, -degree, reshape=True, mode='nearest')
				rot1 = np.expand_dims(cv2.resize(img_rot1,self.target_shape), axis=0)
				img_rot2 = ndimage.rotate(x, degree, reshape=True, mode='nearest')
				rot2 = np.expand_dims(cv2.resize(img_rot2,self.target_shape), axis=0)
				featb5c3 = modelb5c3end.predict(rot1)
				featb5c3_2 = modelb5c3end.predict(rot2)

				try:
					featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]))
				except:
					featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]*featb5c3.shape[2]*featb5c3.shape[3]))
				if i==0:
					featb5c3_ = featb5c3
					featb5c3_ = np.append(featb5c3_, featb5c3_2, axis=0)
					i+=1
				else:
					featb5c3_ = np.append(featb5c3_, featb5c3, axis=0)
					featb5c3_ = np.append(featb5c3_, featb5c3_2, axis=0)
				j+=1        
				s = file.split('/')
				os.makedirs("features/augment/"+s[1]+"/"+s[2],exist_ok=True)
				feature_file_path = "features/augment/"+file[9:-4]+"+20"+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				np.save(feature_file_path, featb5c3 )
				feature_file_path = "features/augment/"+file[9:-4]+"-20"+"block"+str(self.layer)+"_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				np.save(feature_file_path, featb5c3_2 )
			print(str(j) + " features loaded")
		return featb5c3_
	def get_fusion(self):
		w_string = self.weights.split('/')[1].split('.')[0]
		try:
			for file in self.files:
				feature_file_path_1 = "features/"+file[9:-4]+"block1_conv2_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_2 = "features/"+file[9:-4]+"block2_conv2_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_3 = "features/"+file[9:-4]+"block3_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_4 = "features/"+file[9:-4]+"block4_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_5 = "features/"+file[9:-4]+"block5_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_1 = np.load(feature_file_path_1)
				feature_file_2 = np.load(feature_file_path_2)
				feature_file_3 = np.load(feature_file_path_3)
				feature_file_4 = np.load(feature_file_path_4)
				feature_file_5 = np.load(feature_file_path_5)
				try:
					featb1c2_ = np.append(featb1c2_, feature_file_1, axis=0)
					featb2c2_ = np.append(featb2c2_, feature_file_2, axis=0)
					featb3c3_ = np.append(featb3c3_, feature_file_3, axis=0)
					featb4c3_ = np.append(featb4c3_, feature_file_4, axis=0)
					featb5c3_ = np.append(featb5c3_, feature_file_5, axis=0)
				except:
					featb1c2_ = feature_file_1
					featb2c2_ = feature_file_2
					featb3c3_ = feature_file_3
					featb4c3_ = feature_file_4
					featb5c3_ = feature_file_5
		except:
			pass
			if self.path==None:
				raise Exception('no file path')
			if self.weights==None or self.weights=='pretrained':
				model = applications.VGG16(weights='imagenet', include_top=False,input_shape=self.input_shape)
				print('Imagenet loaded')
			else:
				model = applications.VGG16(weights=None, include_top=False,input_shape=self.input_shape)
				model.load_weights(self.weights)
				print('Weights Loaded')
			modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block5_conv3').output)
			modelb4c3 = Model(inputs=model.input, outputs=model.get_layer('block4_conv3').output)
			modelb3c3 = Model(inputs=model.input, outputs=model.get_layer('block3_conv3').output)
			modelb2c2 = Model(inputs=model.input, outputs=model.get_layer('block2_conv2').output)
			modelb1c2 = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)
			print('Model loaded.')
			# Create new Model to add global pooling layer
			modelb5c3end = Sequential()
			modelb4c3end = Sequential()
			modelb3c3end = Sequential()
			modelb2c2end = Sequential()
			modelb1c2end = Sequential()
			if self.pooling=='max':
				#Block5
				modelb5c3end.add(GlobalMaxPooling2D())
				#Block4			
				modelb4c3end.add(GlobalMaxPooling2D())			
				#Block3			
				modelb3c3end.add(GlobalMaxPooling2D())			
				#BLock2			
				modelb2c2end.add(GlobalMaxPooling2D())			
				#Block1			
				modelb1c2end.add(GlobalMaxPooling2D())			
			elif self.pooling=='avg':
				#Block5
				modelb5c3end.add(GlobalAveragePooling2D())
				#Block4
				modelb4c3end.add(GlobalAveragePooling2D())
				#Block3
				modelb3c3end.add(GlobalAveragePooling2D())
				#BLock2
				modelb2c2end.add(GlobalAveragePooling2D())
				#Block1
				modelb1c2end.add(GlobalAveragePooling2D())
			else:
				raise Exception('invalid pooling, avg or max')
			modelb5c3end = Model(inputs=modelb5c3.input, outputs=modelb5c3end(modelb5c3.output))
			modelb4c3end = Model(inputs=modelb4c3.input, outputs=modelb4c3end(modelb4c3.output))
			modelb3c3end = Model(inputs=modelb3c3.input, outputs=modelb3c3end(modelb3c3.output))
			modelb2c2end = Model(inputs=modelb2c2.input, outputs=modelb2c2end(modelb2c2.output))
			modelb1c2end = Model(inputs=modelb1c2.input, outputs=modelb1c2end(modelb1c2.output))
			z = []
			z_val = []
			i=0
			j=0
			for file in self.files:			
				img = image.load_img(file, target_size=self.target_shape)
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)	
				featb1c2 = modelb1c2end.predict(x)
				featb2c2 = modelb2c2end.predict(x)
				featb3c3 = modelb3c3end.predict(x)
				featb4c3 = modelb4c3end.predict(x)
				featb5c3 = modelb5c3end.predict(x)
				try:
					featb1c2 = featb1c2.reshape((1,featb1c2.shape[1]))
					featb2c2 = featb2c2.reshape((1,featb2c2.shape[1]))
					featb3c3 = featb3c3.reshape((1,featb3c3.shape[1]))
					featb4c3 = featb4c3.reshape((1,featb4c3.shape[1]))
					featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]))
				except:
					featb5c3 = featb5c3.reshape((1,featb5c3.shape[1]*featb5c3.shape[2]*featb5c3.shape[3]))
				if i==0:
					featb1c2_ = featb1c2
					featb2c2_ = featb2c2
					featb3c3_ = featb3c3
					featb4c3_ = featb4c3
					featb5c3_ = featb5c3
					i+=1
				else:
					featb1c2_ = np.append(featb1c2_, featb1c2, axis=0)
					featb2c2_ = np.append(featb2c2_, featb2c2, axis=0)
					featb3c3_ = np.append(featb3c3_, featb3c3, axis=0)
					featb4c3_ = np.append(featb4c3_, featb4c3, axis=0)
					featb5c3_ = np.append(featb5c3_, featb5c3, axis=0)
				j+=1
				s = file.split('/')
				os.makedirs("features/"+s[1]+"/"+s[2],exist_ok=True)
				feature_file_path_1 = "features/"+file[9:-4]+"block1_conv2_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_2 = "features/"+file[9:-4]+"block2_conv2_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_3 = "features/"+file[9:-4]+"block3_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_4 = "features/"+file[9:-4]+"block4_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				feature_file_path_5 = "features/"+file[9:-4]+"block5_conv3_"+self.pooling+"_feature_"+w_string+".npy"
				np.save(feature_file_path_1, featb1c2)
				np.save(feature_file_path_2, featb2c2)
				np.save(feature_file_path_3, featb3c3)
				np.save(feature_file_path_4, featb4c3)
				np.save(feature_file_path_5, featb5c3)

			print(str(j) + " features loaded")
			os.makedirs('features/'+self.path,exist_ok=True)
		dict_ = [featb1c2_,featb2c2_,featb3c3_,featb4c3_,featb5c3_]
		out=np.empty((int(dict_[0].shape[0]),0))
		for i in self.layer:
			out = np.append(out,dict_[i-1],axis=1)
		return out
