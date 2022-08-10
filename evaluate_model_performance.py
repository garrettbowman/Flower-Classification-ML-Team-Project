# --------------------- Imports --------------------

import sys
import os

import tensorflow as tf
from tensorflow import keras

import pandas as pd
from sklearn.metrics import classification_report

import numpy as np
import numpy.random as npr

import warnings
warnings.filterwarnings("ignore")


# --------------------- get arguments for import ---------------------

X_test_full = np.load(sys.argv[1])
t_test_full = np.load(sys.argv[2])


print('\n')
print("Dataset Sanity Check!", '\n')
print("The expected form is: 270000 x M, where M is the number of samples", '\n')
print("Your specified data set: ", X_test_full.shape, '\n')

# -------------------- transpose ------------------------------------

X_test_full = X_test_full.T

# -------------------- specify labels for later classification ------

class_names = ['Roses', 'Magnolias', 'Lilies', 'Sunflowers', 'Orchids', 'Marigold', 
               'Hibiscus', 'Firebush', 'Pentas', 'Bougainvillea']


# --------------------- reshape -------------------------------------

# Cast to a TensorFlow tensor and reshaping it as a Mx300x300x3
X_test_rs = tf.constant(X_test_full.reshape((X_test_full.shape[0],300,300,3)), dtype=tf.float32)


# --------------------- Load model from command line args

model = keras.models.load_model(sys.argv[3])

score, acc = model.evaluate(X_test_rs, t_test_full)

print('\n')
print('Test score:', round(score*100,2))
print('Test accuracy:', round(acc*100,2))
print('\n')

# -------------------- performance testing ------------------------

y_test = np.argmax(model.predict(X_test_rs), axis=1)

print('Test classification report')
report = classification_report(t_test_full, y_test, target_names=class_names, output_dict=True)
print(classification_report(t_test_full, y_test, target_names=class_names))
print('\n')

if not os.path.exists('csv'):
    os.mkdir('csv')

report_df = pd.DataFrame(report)
report_df.to_csv('csv/classification_report.csv')

print("Classification report saved to 'csv/classification_report.csv'")
