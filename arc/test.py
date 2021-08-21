"""
Testing Session
Run as:
    python test.py <data_path>
"""

import os
import sys
import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# from seaborn import heatmap

from utility_functions.collect_dataset import create_data_label_lists
from utility_functions.batch_generator import BatchGenerator
from utility_functions.cnn_model import cnn_model
# from utility_functions.item_names import ITEM_NAMES


def test_model(net_name, net_configuration, snapshot, data_folder, batch_size=16):
    # Configurations
    snapshot_name = '{}_epoch_{}_loss_{}.hdf5'.format(
        net_name, snapshot['epoch'], snapshot['loss']
    )
    snapshot_file = os.path.join(snapshot['dir'], net_name, snapshot_name)

    test_data_list, test_label_list = create_data_label_lists(data_folder, imgs=range(280, 310+1))
    test_generator = BatchGenerator(test_data_list, test_label_list, batch_size=batch_size, aug_flag=False)

    model = cnn_model(net_configuration)
    model.load_weights(snapshot_file)

    predictions = model.predict_generator(generator=test_generator, verbose=1, max_queue_size=8, workers=4,
                                          use_multiprocessing=True)

    predictions_argmax = np.argmax(np.asarray(predictions, dtype=np.uint16), axis=-1)
    label_list_argmax = np.asarray(test_label_list, dtype=np.uint16)
    correct_preds = np.sum(predictions_argmax == label_list_argmax)

    # Overall Accuracy
    print('Accuracy: {}/{}={}'.format(correct_preds, len(label_list_argmax), correct_preds/len(label_list_argmax)))

    # Per-class Accuracy
    #acc_func = lambda y_true, y_pred:np.sum(y_true==y_pred)/len(y_true)
    #per_class_acc = [acc_func(label_list_argmax==i, predictions_argmax==i) for i in range(100)]
    #per_class_acc_string = [str(i) for i in per_class_acc]
    #print('Per-Class Accuracy:')
    #print('\n'.join(per_class_acc_string))

    # Classification Report
    #report = classification_report(label_list_argmax, predictions_argmax)
    #with open(net_name+'_report.txt', 'wt') as report_file:
    #    report_file.write(report)
    # Confusion Matrix
    #cm = confusion_matrix(label_list_argmax, predictions_argmax) # labels
    #ax = heatmap(cm, xticklabels=ITEM_NAMES, yticklabels=ITEM_NAMES)
    #plt.show()


if __name__ == '__main__':
    assert len(sys.argv) == 2
    data_path = sys.argv[1]

    test_args = {
        'net_name': 'arc_cnn',
        'net_configuration': {
            'BatchNorm': True,
            'Activation': 'prelu',
            'Regularization': 0.01,
            'Dropout': 0.1,
        },
        'snapshot': {'dir': 'models', 'epoch': 29, 'loss': 0.71},
        'data_folder': data_path,
        'batch_size': 16,
    }

    test_model(**test_args)
