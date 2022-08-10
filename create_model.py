# --------------------- Imports --------------------

import sys
import os

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import numpy as np

import warnings
warnings.filterwarnings("ignore")


# --------------------- get arguments for import ---------------------

X_train_full = np.load(sys.argv[1])
t_train_full = np.load(sys.argv[2])


# --------------------- preprocessing -------------------------------

#This removes the bad images.
x = X_train_full[:,:506]
x1 = X_train_full[:,507:561]
x2 = X_train_full[:,562:694]
x3 = X_train_full[:,695:1057]
x4 = X_train_full[:,1058:]
X_train_full = np.column_stack((x,x1,x2,x3,x4))

y = t_train_full[:506]
y1 = t_train_full[507:561]
y2 = t_train_full[562:694]
y3 = t_train_full[695:1057]
y4 = t_train_full[1058:]
t_train_full = np.row_stack((y[:,np.newaxis],y1[:,np.newaxis],y2[:,np.newaxis],y3[:,np.newaxis],y4[:,np.newaxis]))
t_train_full = np.squeeze(t_train_full, 1)

# -------------------- transpose ------------------------------------

X_train_full = X_train_full.T

# -------------------- split into training and test sets ------------

# Training and Test sets
X_training, X_test, t_training, t_test = train_test_split(X_train_full,
                                                 t_train_full,
                                                 test_size=0.1,
                                                 stratify=t_train_full,
                                                 shuffle=True)

# Train and validation sets
X_train, X_val, t_train, t_val = train_test_split(X_training, t_training,
                                                 test_size=0.11, stratify=t_training,
                                                shuffle=True)


class_names = ['Roses', 'Magnolias', 'Lilies', 'Sunflowers', 'Orchids', 'Marigold', 
               'Hibiscus', 'Firebush', 'Pentas', 'Bougainvillea']


# Cast to a TensorFlow tensor and reshapping it as a 300x300x3
X_train_rs = tf.constant(X_train.reshape((X_train.shape[0],300,300,3)), dtype=tf.float32)


X_val_rs = tf.constant(X_val.reshape((X_val.shape[0],300,300,3)), dtype=tf.float32)


X_test_rs = tf.constant(X_test.reshape((X_test.shape[0],300,300,3)), dtype=tf.float32)


# ------------------- transfer learning to train the model ---------

#Basically can just just change all the values to whatever to see whats best
def build_augmenter(ORG_IMG_SIZE=300, IMG_SIZE=300,CENT_CROP=.8,ZOOM_FACT=.1,RAND_ROT=.3): 
#rand_rot = .3
#rand_zoom = .1

    data_augmentation = tf.keras.Sequential([
        #tf.keras.layers.CenterCrop(int(ORG_IMG_SIZE*CENT_CROP),int(ORG_IMG_SIZE*CENT_CROP)),
        tf.keras.layers.RandomZoom(height_factor=(-ZOOM_FACT,ZOOM_FACT),width_factor=(-ZOOM_FACT,ZOOM_FACT)),
        tf.keras.layers.RandomRotation(RAND_ROT),
        #tf.keras.layers.Resizing(IMG_SIZE,IMG_SIZE),
        tf.keras.layers.Rescaling(1./255),
        #tf.keras.layers.RandomFlip('horizontal_and_vertical')
    ])
    
    return data_augmentation


data_augmentation = build_augmenter()

base_model = keras.applications.Xception(
    weights='imagenet', # loads the weights pre-trained using imagenet dataset
    input_shape=(300,300,3),
    include_top = False
)


# freeze all the weights from this pre-trained model
base_model.trainable = False


# ----------------------- Model layers ----------------------------
#100/50/10 dense
#.2/.2/.4 drop best 99.39

#This is set to best model we found.
inputs = keras.Input(shape=(300,300,3))
data_augmented = data_augmentation(inputs)

x = base_model(data_augmented, training=False) 

# Flatten the output of the pre-trained model and feed it to a dense layer
y = keras.layers.GlobalAveragePooling2D()(x)

y = keras.layers.Dense(100, activation='selu')(y)
y = keras.layers.Dropout(.2)(y)
y = keras.layers.Dense(50, activation='selu')(y)
y = keras.layers.Dropout(.2)(y)
y = keras.layers.Dense(10, activation='softmax')(y)

outputs = keras.layers.Dropout(.4)(y)

# Wrapping the model
model = keras.Model(inputs, outputs)


#This is set to best learning rate we found
model.compile(loss='sparse_categorical_crossentropy',
             metrics=['accuracy'],
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))


#Just ran 10000 since we are using patience found 100 was a good amount.
model.fit(X_train_rs, t_train, epochs=10000,
         validation_data=(X_val_rs, t_val),
         callbacks=[keras.callbacks.EarlyStopping(patience=100),
                   keras.callbacks.ModelCheckpoint("image_model.h5", save_best_only=True)])

print('\n')
print("Trained model was successfully saved to 'image_model.h5'")


if sys.argv[3] == '--debug':
    # Saves the test dataset and its labels (derived from the training set) to /debug for evaluating accuracy
    
    if not os.path.exists('debug'):
        os.mkdir('debug')

    np.save('debug/X_test_from_training', X_test.T)
    np.save('debug/X_test_from_training_labels', t_test)
    print('\n')
    print("Derived test set and its labels were saved to '/debug' successfully")
