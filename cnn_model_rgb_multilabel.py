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


def build_model(filters1=32,
                ksize1=(5, 5),
                activation1='relu',
                pool_size1=(2, 2),
                stride1=2,
                filters2=64,
                ksize2=(5, 5),
                activation2='relu',
                pool_size2=(2, 2),
                stride2=2,
                dense1=8,
                activation_dense1='relu',
                activation_dense2='softmax',
                general_optimizer='adam',
                general_loss='categorical_crossentropy',
                general_metrics=['accuracy', 'mae', 'mse', ]
                ):

    model = Sequential()  # Create the architecture
    model.add(Conv2D(filters=filters1,
                     kernel_size=ksize1,
                     activation=activation1,
                     input_shape=(217, 383, 3)
                     )
              )
    model.add(MaxPooling2D(pool_size=pool_size1, strides=stride1))
    model.add(Conv2D(filters=filters2,
                     kernel_size=ksize2,
                     activation=activation2,
                     input_shape=(217, 383, 3)
                     )
              )
    model.add(MaxPooling2D(pool_size=pool_size2, strides=stride2))
    model.add(Flatten())
    # a layer with 1000 neurons and activation function ReLu
    model.add(Dense(dense1, activation=activation_dense1))
    # a layer with 2 output neurons 1 for each label using softmax activation f
    model.add(Dense(5, activation=activation_dense2))

    model.compile(loss=general_loss,
                  optimizer=general_optimizer,
                  metrics=general_metrics)
    return model


def train_model(model,
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

    hist = model.fit(x_train,
                     y_train_one_hot,
                     batch_size=batch,
                     epochs=epochs,
                     validation_split=0.3,
                     # validation_data=(x_test, y_test_one_hot),
                     callbacks=callbacks,
                     )
    return hist


model = build_model()

model.summary()

history = train_model(model, tboard=True, ckpt=True)
