from datetime import datetime
import os
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import ModelCheckpoint
import talos

dir_labels = os.listdir("images/rgb_files_multilabel/")

whole = pd.DataFrame([], columns=['filename', 'label'])

for dl in dir_labels:
    label_list = os.listdir("images/rgb_files_multilabel/" + dl)
    for l in label_list:
        whole = whole.append({'filename': l, 'label': dl}, ignore_index=True)
        pass
    pass


whole_name = [t[:34] for t in whole["filename"]]

whole_satno = [t[:4][-2:] for t in whole["filename"]]

whole["img_name"] = whole_name
whole["sat_no"] = whole_satno

whole = whole.sample(frac=1, random_state=1).reset_index(drop=True)

# train_size = len(whole)
train_size = 100

x_train_df = whole[0:train_size]['filename']
# x_test_df = whole[151:645]['filename']

y_train_df = whole[0:train_size]['label']
# y_train_df.astype('category').describe()
y_train_df = y_train_df.astype('category')

# y_test_df = whole[151:645]['label']
# y_test_df.astype('category').describe()
# y_test_df = y_test_df.astype('category')

y_train = np.array(y_train_df)
# y_test = np.array(y_test_df)

y_train.shape  # O ERRO ESTA AQUI
y_train = np.reshape(y_train, (train_size, 1))
# y_test.shape  # O ERRO ESTA AQUI
# y_test = np.reshape(y_test, (494, 1))

no_files = len(x_train_df)

x_train = np.empty((no_files, 217, 383, 3))
j = 0

for i in x_train_df:
    image = np.array(Image.open("images/rgb_files_resized/" + i))
    x_train[j] = image
    j = j+1


# no_files = len(x_test_df)
#
# x_test = np.empty((no_files, 217, 383, 3))
# j = 0
#
# for i in x_test_df:
#     image = np.array(Image.open("rgb_files_resized/" + i))
#     x_test[j] = image
#     j = j+1


y_train_one_hot = to_categorical(y_train_df.factorize()[0])
# y_test_one_hot = to_categorical(y_test_df.factorize()[0])


def build_model(params):

    model = Sequential()  # Create the architecture
    model.add(Conv2D(filters=params['filters1'],
                     kernel_size=params['ksize1'],
                     activation=params['activation1'],
                     input_shape=(217, 383, 3)
                     )
              )
    model.add(MaxPooling2D(pool_size=params['pool_size1'], strides=params['stride1']))
    model.add(Conv2D(filters=params['filters2'],
                     kernel_size=params['ksize2'],
                     activation=params['activation2'],
                     input_shape=(217, 383, 3)
                     )
              )
    model.add(MaxPooling2D(pool_size=params['pool_size2'], strides=params['stride2']))
    model.add(Flatten())
    # a layer with 1000 neurons and activation function ReLu
    model.add(Dense(params['dense1'], activation=params['activation_dense1']))
    # a layer with 2 output neurons 1 for each label using softmax activation f
    model.add(Dense(5, activation=params['activation_dense2']))

    model.compile(loss=params['general_loss'],
                  optimizer=params['general_optimizer'],
                  metrics=['accuracy', 'mae', 'mse', ])
    return model


def train_model(x,
                y,
                model,
                name="noname",
                tboard=False,
                ckpt=False,
                epochs='10',
                batch='3'):

    callbacks = []

    if tboard is True:
        logdir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S-") + name
        if not os.path.exists('logs'):
            os.makedirs('logs')

        os.mkdir(logdir)
        logdir = os.path.join(logdir)

        tensorboard_callback = TensorBoard(log_dir=logdir,
                                           histogram_freq=1,
                                           profile_batch=100000000)

        callbacks.append(tensorboard_callback)
        pass

    if ckpt is True:

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        ckpf = os.path.join('checkpoints/',
                            datetime.now().strftime("%Y%m%d-%H%M%S-"),
                            name,
                            '.hdf5')

        checkpointer = ModelCheckpoint(filepath=ckpf,
                                       verbose=1,
                                       save_best_only=True)

        callbacks.append(checkpointer)
        pass

    hist = model.fit(x,
                     y,
                     batch_size=batch,
                     epochs=epochs,
                     validation_split=0.3,
                     # validation_data=(x_test, y_test_one_hot),
                     callbacks=callbacks,
                     )
    return hist


# model = build_model()

def talos_model(params):
    model = build_model(params)
    history = train_model(x=x_train,
                y=y_train_one_hot,
                model=model,
                name="noname",
                tboard=False,
                ckpt=False,
                epochs='1',
                batch='3')
    return history, model

# p = {
#     'filters1' : [12,24,32],
#     'ksize1' : (5, 5),
#     'activation1' : 'relu',
#     'pool_size1' : (2, 2),
#     'stride1' : 2,
#     'filters2' : 64,
#     'ksize2' : (5, 5),
#     'activation2' : 'relu',
#     'pool_size2' : (2, 2),
#     'stride2' : 2,
#     'dense1' : 8,
#     'activation_dense1' : 'relu',
#     'activation_dense2' : 'softmax',
#     'general_optimizer' : 'adam',
#     'general_loss' : 'categorical_crossentropy',
#     }

p = {
    'filters1' : [12,24,32],
    'ksize1' : [(5, 5)],
    'activation1' : ['relu'],
    'pool_size1' : [(2, 2)],
    'stride1' : [2],
    'filters2' : [64],
    'ksize2' : [(5, 5)],
    'activation2' : ['relu'],
    'pool_size2' : [(2, 2)],
    'stride2' : [2],
    'dense1' : [8],
    'activation_dense1' : ['relu'],
    'activation_dense2' : ['softmax'],
    'general_optimizer' : ['adam'],
    'general_loss' : ['categorical_crossentropy'],
    }

t=None
t = talos.Scan(x=x_train, y=y_train_one_hot, params=p, model=talos_model, experiment_name='talos_model')

# model.summary()

# history = train_model(model, tboard=True, ckpt=True)
