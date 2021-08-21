"""
Training Session
Run as:
    python train.py <data_path>
"""

import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (CSVLogger, ModelCheckpoint, TensorBoard, LearningRateScheduler,
                                        ReduceLROnPlateau)

from utility_functions.collect_dataset import create_data_label_lists
from utility_functions.batch_generator import BatchGenerator
from utility_functions.cnn_model import cnn_model

np.random.seed(32)
# tf.set_random_seed(64)    # tf 1.x
tf.random.set_seed(64)      # tf 2.x


class LRTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def exponential_schedule(epoch, current_learning_rate):
    if epoch < 20:
        return current_learning_rate*0.96
    else:
        return current_learning_rate*0.75


def train_model(net_configuration, data_folder, initial_learning_rate=1e-3, batch_size=16, augmentation_flag=True,
                epochs=100, net_name='arc_cnn'):

    print('**********************************************************')
    print(f'* Initiating Training Session for Model: \'{net_name}\'')
    print('**********************************************************')

    assert os.path.isdir(data_folder), f"[Error] Directory `{data_folder}` does not exist."
    net_folder = os.path.join('./models', net_name)

    # Data
    train_data_list, train_label_list = create_data_label_lists(data_folder, imgs=range(1, 217+1))
    val_data_list, val_label_list = create_data_label_lists(data_folder, imgs=range(218, 279+1))

    train_generator = BatchGenerator(train_data_list, train_label_list, batch_size=batch_size,
                                     aug_flag=augmentation_flag)
    val_generator = BatchGenerator(val_data_list, val_label_list, batch_size=batch_size,
                                   aug_flag=augmentation_flag)

    # Network
    model = cnn_model(net_configuration)
    optimizer = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0., amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Model Summary:')
    for i, layer in enumerate(model.layers):
        print ("Layer", i, "\t", layer.name, "\t\t", layer.input_shape, "\t", layer.output_shape)

    # Callbacks
    ckpt_save_name = os.path.join(net_folder, net_name+'_epoch_{epoch:02d}_loss_{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(ckpt_save_name, monitor='val_loss', verbose=1, save_best_only=False)
    tensorboard = LRTensorBoard(log_dir=net_folder)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-10, verbose=1)
    lr_scheduler = LearningRateScheduler(exponential_schedule, verbose=1)
    log_csv = CSVLogger(net_folder+'/'+net_name+ '.csv', separator='\t', append=False)
    callbacks = [checkpoint, reduce_lr, lr_scheduler, tensorboard, log_csv]

    # Training
    model.fit_generator(generator=train_generator, epochs=epochs, verbose=1, validation_data=val_generator,
                        callbacks=callbacks, max_queue_size=16, workers=4, use_multiprocessing=True)

    print('**********************************************************')
    print(f'* Completed Training Session for Model: \'{net_name}\'')
    print('**********************************************************')


if __name__ == '__main__':
    assert len(sys.argv) == 2
    data_path = sys.argv[1]

    train_args = {
        # required args:
        'net_configuration': {
            'BatchNorm': True,
            'Activation': 'prelu',
            'Regularization': 0.01,
            'Dropout': 0.1,
        },
        'data_folder': data_path,
        # optional args:
        'initial_learning_rate': 1e-3,
        'batch_size': 16,
        'augmentation_flag': False,
        'epochs': 100,
        'net_name': 'arc_cnn',
    }

    train_model(**train_args)
