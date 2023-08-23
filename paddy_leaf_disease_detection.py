import os
# from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""## EXPLORATORY DATA ANALYSIS"""

from google.colab  import drive
drive.mount('/content/drive')

data_path='/content/drive/MyDrive/Colab Notebooks/paddy-disease-classification/train_images'

directory_names = os.listdir(data_path)

for directory in directory_names:
    dir_path = os.path.join(data_path, directory)
    if os.path.isdir(dir_path):
        files = os.listdir(dir_path)
        print(f"{len(files)} \t {directory}")
        for file in files:
          file_path = os.path.join(dir_path, file)
          if file.endswith('.csv'):
              data_path = pd.read_csv(file_path)
              print(f"\nData in {file}:\n{data_path.head()}\n")

csv_file_path = '/content/drive/MyDrive/Colab Notebooks/paddy-disease-classification/train.csv'
meta = pd.read_csv(os.path.join(data_path, csv_file_path))
meta.head()

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set the main directory path
main_dir = '/content/drive/MyDrive/Colab Notebooks/paddy-disease-classification'

# Navigate to the train images directory
train_dir = os.path.join(main_dir, 'train_images')

# List all subdirectories within the train images directory
subdirectories = [subdir for subdir in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, subdir))]

# Set up an 8x10 grid of subplots for displaying images
plt.figure(figsize=(20, 40))
for i, subdir in enumerate(subdirectories):
    subdir_path = os.path.join(train_dir, subdir)

    # List all image files in the current subdirectory
    image_files = [file for file in os.listdir(subdir_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Display up to 8 images from the current subdirectory
    for j in range(min(8, len(image_files))):  # Display up to 8 images
        if (i * 8 + j) >= 80:
            break  # Break loop if we have displayed 80 images
        image_path = os.path.join(subdir_path, image_files[j])
        img = mpimg.imread(image_path)

        # Add subplot to the grid
        plt.subplot(10, 8, i * 8 + j + 1)
        plt.imshow(img)
        plt.title(f'Subdirectory: {subdir}\nImage: {image_files[j]}')
        plt.axis('off')

plt.tight_layout()
plt.show()

# ## distribution of labels
meta.groupby('label')['label'].count().sort_values().plot.barh(figsize=(8,5))

# ## distribution of labels in percentage
(meta.groupby('label')['label'].count().sort_values() / meta.shape[0] * 100).plot.barh(figsize=(8,5))

fig,ax = plt.subplots(figsize = (10, 6))
sns.countplot(y = 'label',
              data = meta,
              order = meta['label'].value_counts().index)
plt.show()
print(meta["label"].value_counts())

fig,ax = plt.subplots(figsize = (10, 6))
sns.countplot(y = 'variety',
              data = meta,
              order = meta['variety'].value_counts().index)
plt.show()
print(meta["variety"].value_counts())

fig,ax = plt.subplots(figsize = (10, 6))
sns.countplot(y = 'age',
              data = meta,
              order = meta['age'].value_counts().index)
plt.show()
print(meta["age"].value_counts())

sns.distplot(meta["age"])

fig,ax = plt.subplots(figsize = (12, 6))
ax.set_xticklabels(meta["label"].index, rotation = 20)
ax.set_title("Labels age distribution")
sns.violinplot(x = "label", y = "age", data = meta)
plt.show()

fig,ax = plt.subplots(figsize = (12, 6))
ax.set_xticklabels(meta["label"].index, rotation = 30)
sns.countplot(x = "label", hue = "variety", data = meta)

"""## 2. Verify the files in the train and test directories"""

import glob
from pathlib import Path

##For Google Colab
train_path = '/content/drive/MyDrive/Colab Notebooks/paddy-disease-classification/train_images'
test_path  = '/content/drive/MyDrive/Colab Notebooks/paddy-disease-classification/test_images'

print('training set')
for filepath in glob.glob(train_path + '/*/'):
  files = glob.glob(filepath + '*')
  print(f"{len(files)} \t {Path(filepath).name}")

print('testing set')
for filepath in glob.glob(test_path + '/*/'):
  files = glob.glob(filepath + '*')
  print(f"{len(files)} \t {Path(filepath).name}")

"""## Import the necesary python libraries"""

import os
import random
from os import listdir
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import matplotlib.pyplot as plt

SEED = 1234
def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

"""## Define the necessary constants"""

SEED = 1234
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
width = 256
height = 256
depth = 3
n_classes = len(glob.glob(train_path + '/*/'))
print(n_classes)

"""## 3. Dataset preparation (train, validate, and test sets)

### 3a. Setup ImageDataGenerator with different image transformation options to generate diverse training samples.
"""

image_datagen = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   rotation_range=5,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   channel_shift_range=0.,
                                   fill_mode='nearest',
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1./255,
                                   validation_split=0.2)

"""### 3b. Next, let's configure the train, validate, and test data generators usingflow_from_directory"""

train_generator = image_datagen.flow_from_directory(
    directory = train_path,
    subset='training',
    target_size=(256, 256),
color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True,
seed=SEED)
valid_generator = image_datagen.flow_from_directory(
    directory=train_path,
    subset='validation',
    target_size=(256, 256),
color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True,
    seed=SEED)
print(train_generator.class_indices)
print(train_generator.samples)

"""# 4. CNN model"""

def get_model():
  model = Sequential()
  inputShape = (height, width, depth)
  chanDim = -1
  print(backend.image_data_format())
  if backend.image_data_format() == "channels_first":
          inputShape = (depth, height, width)
          chanDim = 1
  model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(64, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(Conv2D(128, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(n_classes))
  model.add(Activation("softmax"))
  # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
  opt = Adam(learning_rate=INIT_LR)
  # distribution
  model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
  return model

model = get_model()
plot_model(model, 'cnn-model.png', show_shapes=True)

"""# 5. Model training

### 5a. (Optional) Setup and configure training checkpoints, early stopping, and ploting call backs to lively visualize the training
"""

try:
  import livelossplot
except:
  !pip install livelossplot

from livelossplot.inputs.keras import PlotLossesCallback

plot_loss = PlotLossesCallback()

# ModelCheckpoint callback - save best weights
checkpoint = ModelCheckpoint(filepath='paddy-doctor-small-cnn.best.hdf5',
                             save_best_only=True,
                             verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

%%time
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = STEP_SIZE_TRAIN,
                              validation_steps = STEP_SIZE_VALID,
                              validation_data = valid_generator,
                              callbacks=[checkpoint, early_stop, plot_loss],
                              epochs=EPOCHS,
verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()