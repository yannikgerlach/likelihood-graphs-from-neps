import warnings

import arrow
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning

from april import Dataset, fs
from april.alignments.binet import BINet

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def train():
    datasets = ['medium-0.3-1', 'p2p-0.3-1', 'paper-0.3-1', 'small-0.3-1', 'wide-0.3-1', 'huge-0.3-1']

    for dataset_name in datasets:

        dataset = Dataset(dataset_name, use_event_attributes=True, use_case_attributes=False)
        x, y = dataset.features, dataset.targets

        number_attributes, number_epochs, batch_size = dataset.num_attributes, 1000, 50

        # for two attributes, v2 and v3 are the same
        binet_versions = [(0, 0), (1, 0)]
        if number_attributes > 2:
            binet_versions.append((1, 1))

        # for different versions of the binet
        for (present_activity, present_attribute) in binet_versions:
            start_time = arrow.now()

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01)

            binet = BINet(dataset, use_case_attributes=False, use_event_attributes=True,
                          use_present_activity=present_activity, use_present_attributes=present_attribute)
            binet.fit(x=x, y=y, batch_size=batch_size, epochs=number_epochs, validation_split=0.1, callbacks=[callback])

            binet.save_weights(str(
                fs.MODEL_DIR / f'{dataset_name}_{binet.name}{present_activity}{present_attribute}_{start_time.format(fs.DATE_FORMAT)}.h5'))

            tf.keras.backend.clear_session()


if __name__ == '__main__':
    train()
