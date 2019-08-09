import os

from keras import layers, optimizers
from keras.engine.saving import load_model
from keras.initializers import RandomUniform
from keras.preprocessing import image
import numpy as np
from keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, MaxPool2D

from GUI import GraphicUserInterface

from keras.utils import plot_model


def plt_modle(model_hist):
    acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()


def train(flower_path):
    # Split images into Training and Validation Sets (20%)

    train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2,
                               width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

    img_size = 128
    batch_size = 20
    t_steps = 3462 / batch_size
    v_steps = 861 / batch_size
    classes = 5
    train_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                          class_mode='categorical', subset='training')
    valid_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                          class_mode='categorical', subset='validation')

    # Model

    model = models.Sequential()

    # use model.add() to add any layers
    initializer = RandomUniform(0.8, 1)
    config = initializer.get_config()
    initializer = RandomUniform.from_config(config)

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 3),
                     name='conv_1'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='maxpool_1'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='maxpool_2'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='maxpool_3'))
    # model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='maxpool_4'))
    # model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='maxpool_5'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    # last layer should be with softmax activation function
    model.add(layers.Dense(classes, activation='softmax'))

    # fill optimizer argument using one of keras.optimizers.
    optimizer = 'adam'

    # fill loss argument using keras.losses.
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=27, validation_data=valid_gen,
                                     validation_steps=v_steps)
    model.save('flowers_model.h5')
    plt_modle(model_hist)


class ModelController:

    def __init__(self, model):
        self.model = model
        self.flowers_dict = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

    def predict(self, imgs_path):
        # dimensions of our images
        img_width, img_height = 128, 128
        results = []
        imgs = os.listdir(imgs_path)
        for photo in imgs:
            img = image.load_img(imgs_path + '\\' + photo, target_size=(img_width, img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            res = self.model.predict(x)[0]
            i = 0
            for num in res:
                # print (num)
                if np.equal(num, 1.0):
                    results.append([photo, self.flowers_dict[i]])
                    break
                i += 1
        with open("results.csv", 'w') as f:
            f.write('Image,Classification\n')
            for res in results:
                f.write(res[0] + ',' + res[1] + '\n')
        return results

    def load_model(self):
        return load_model('flowers_model.h5')

    def load_model_by_path(self,model_path):
        return load_model(model_path)


if __name__ == '__main__':
    #os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'
    model = load_model('flowers_model.h5')
    model_controller = ModelController(model)
    #plot_model(model, to_file='graph.png')
    gui = GraphicUserInterface(model_controller)

