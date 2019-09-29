from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import regularizers

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def precision(y_true,y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return c1/c2
  
def recall(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    return c1/c3

def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)

def categorical_squared_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.square(K.maximum(0., neg - pos + 1.)) 

def custom_hinge(y_true,y_pred):
    return K.mean(K.square(K.maximum(0., 1. - y_true * y_pred)))   

# dimensions of our images.
img_width, img_height = 224, 224
train_data_dir = 'datasets/pleo224'
nb_train_samples = 112
classes=7
epochs = 300
batch_size = 16
from datetime import datetime
!pip install -q tf-nightly-2.0-preview
%load_ext tensorboard
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(224, 224, 3))

## Freeze / un-freeze layers
'''for layer in model.layers:
	layer.trainable = False

for layer in model.layers[:12]:
	layer.trainable = True
'''
#model.summary()
#modelb5c3 = Model(inputs=model.input, outputs=model.get_layer('block5_conv3').output)

print('Model loaded.')
'''from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(224, 224, 3))

# a layer instance is callable on a tensor, and returns a tensor
x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
print(x.shape)
predictions = Dense(classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)

# This creates a model that includes
# the Input layer and three Dense layers
model2 = Model(inputs=inputs, outputs=predictions)
model2.compile(optimizer='adadelta',
              loss='categorical_hinge',
              metrics=['accuracy'])
model2.summary()'''
'''model2 = Sequential()
#model2.add(Flatten(input_shape=model.output_shape[1:]))
model2.add(GlobalAveragePooling2D())
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(classes, activation='softmax'))
model2 = Model(inputs=model.input, outputs=model2(model.get_layer('block5_pool').output))
model2.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy', precision, recall, f1_score])'''


model2 = Sequential()
#model2.add(Flatten(input_shape=model.output_shape[1:]))
model2.add(GlobalAveragePooling2D())
model2.add(Dense(64, activation='relu'))
model2.add(Dense(classes, kernel_regularizer=regularizers.l2(0.01)))
model2.add(Activation('linear'))
model2 = Model(inputs=model.input, outputs=model2(model.get_layer('block5_pool').output))
model2.compile(loss=custom_hinge,
              optimizer='adam',
              metrics=['accuracy'])
model2.summary()
'''model2.add(GlobalAveragePooling2D())

#model2.add(Flatten())
model2.add(Dense(classes, kernel_regularizer=regularizers.l2(0.01)))
model2 = Model(input=modelb5c3.input, output=model2(modelb5c3.output))


model2.compile(loss=categorical_squared_hinge,
              optimizer='adadelta',
              metrics=['accuracy'])'''

# Pure SVM implementation https://stackoverflow.com/questions/54414392/convert-sklearn-svm-svc-classifier-to-keras-implementation

'''model2 = Sequential()
#model2.add(GlobalAveragePooling2D())
model2.add(Flatten(input_shape=model.output_shape[1:]))
#model2.add(Dense(64, activation='relu'))
model2.add(Dense(classes, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
model2 = Model(inputs=model.input, outputs=model2(model.get_layer('block5_pool').output))
model2.compile(loss=categorical_squared_hinge,
              optimizer='adadelta',
              metrics=['accuracy'])'''

#model2.summary()



# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #shear_range=0.2,
    #zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2)

datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# fine-tune the model
model2.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    shuffle=True,
    callbacks=[tensorboard_callback])
t = model2.evaluate_generator(train_generator,steps=validation_generator.samples // batch_size)
print(t)
curr = 'test10'
model2.save_weights('model224nobg_'+curr+'.h5')
model.save_weights('model224nobg-body_'+curr+'.h5')
model2.save('model224nobg-pleo_'+curr)