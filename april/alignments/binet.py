#  Copyright 2019 Timo Nolle
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  ==============================================================================

import numpy as np
import tensorflow as tf

from april import Dataset
from april.enums import AttributeType
from april.enums import FeatureType


class AnomalyDetectionResult(object):
    def __init__(self,
                 scores,
                 predictions=None,
                 attentions=None,
                 embeddings=None,
                 scores_backward=None,
                 predictions_backward=None,
                 attentions_backward=None,
                 attentions_raw=None):
        self.scores_forward = scores
        self.scores_backward = scores_backward

        self.predictions = predictions
        self.predictions_backward = predictions_backward

        self.attentions = attentions
        self.attentions_backward = attentions_backward
        self.attentions_raw = attentions_raw

        self.embeddings = embeddings

    @property
    def scores(self):
        return self.scores_forward

    @staticmethod
    def minmax_normalize(scores):
        return (scores - scores.min()) / (scores.max() - scores.min())


class BINet(tf.keras.Model):
    abbreviation = 'binet'
    name = 'BINet'

    def __init__(self,
                 dataset,
                 latent_dim=None,
                 use_case_attributes=None,
                 use_event_attributes=None,
                 use_present_activity=None,
                 use_present_attributes=None,
                 use_attention=None):
        super(BINet, self).__init__()

        # Validate parameters
        if latent_dim is None:
            latent_dim = min(int(dataset.max_len * 10), 256)
        if use_event_attributes and dataset.num_attributes == 1:
            use_event_attributes = False
            use_case_attributes = False
        if use_present_activity and dataset.num_attributes == 1:
            use_present_activity = False
        if use_present_attributes and dataset.num_attributes == 1:
            use_present_attributes = False

        # Parameters
        self.latent_dim = latent_dim
        self.use_case_attributes = use_case_attributes
        self.use_event_attributes = use_event_attributes
        self.use_present_activity = use_present_activity
        self.use_present_attributes = use_present_attributes
        self.use_attention = use_attention

        # Single layers
        self.fc = None
        if self.use_case_attributes:
            self.fc = tf.keras.Sequential([
                tf.keras.layers.Dense(latent_dim // 8),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(latent_dim, activation='linear')
            ])

        self.rnn = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)

        # Layer lists
        self.fc_inputs = []
        self.rnn_inputs = []
        self.outs = []

        inputs = zip(dataset.attribute_dims, dataset.attribute_keys, dataset.attribute_types, dataset.feature_types)
        for dim, key, t, feature_type in inputs:
            if t == AttributeType.CATEGORICAL:
                voc_size = int(dim + 1)  # we start at 1, 0 is padding
                emb_dim = np.clip(voc_size // 10, 2, 10)
                embed = tf.keras.layers.Embedding(input_dim=voc_size, output_dim=emb_dim, mask_zero=True)
            else:
                embed = tf.keras.layers.Dense(1, activation='linear')

            if feature_type == FeatureType.CASE:
                self.fc_inputs.append(embed)
            else:
                self.rnn_inputs.append(embed)
                out = tf.keras.layers.Dense(dim + 1, activation='softmax')
                self.outs.append(out)

        self.compile(tf.keras.optimizers.Adam(), 'sparse_categorical_crossentropy')
        self.dataset = dataset

    def call(self, inputs, training=False, return_state=False, initial_state=None):

        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            inputs = [inputs]

        split = len(self.rnn_inputs)

        rnn_x = inputs[:split]
        fc_x = inputs[split:]

        fc_embeddings = []
        for x, input_layer in zip(fc_x, self.fc_inputs):
            if isinstance(input_layer, tf.keras.layers.Dense):
                x = x[:, None]
            x = input_layer(x)
            fc_embeddings.append(x)

        if len(fc_embeddings) > 0:
            if len(fc_embeddings) > 1:
                fc_embeddings = tf.concat(fc_embeddings, axis=-1)
            else:
                fc_embeddings = fc_embeddings[0]

        fc_output = None
        if not isinstance(fc_embeddings, list):
            fc_output = self.fc(fc_embeddings)

        rnn_embeddings = []

        for x, input_layer in zip(rnn_x, self.rnn_inputs):
            x = input_layer(x)
            rnn_embeddings.append(x)

        length_embedding_activity = rnn_embeddings[0].shape[-1]

        if len(rnn_embeddings) > 0:
            if len(rnn_embeddings) > 1:
                rnn_embeddings = tf.concat(rnn_embeddings, axis=-1)
            else:
                rnn_embeddings = rnn_embeddings[0]

        if initial_state is not None:
            rnn, h = self.rnn(rnn_embeddings, initial_state=initial_state)
        elif fc_output is not None:
            if len(fc_output.shape) == 3:
                fc_output = fc_output[:, 0]
            rnn, h = self.rnn(rnn_embeddings, initial_state=fc_output)
        else:
            rnn, h = self.rnn(rnn_embeddings)

        outputs = []
        for i, out in enumerate(self.outs):
            x = rnn
            if i > 0:
                if self.use_present_attributes:
                    #  usually, append all other activity values
                    #  however, the graph generation algorithm is sequential
                    #  thus, we need give it access to only the previous attributes
                    x = tf.concat([x, tf.pad(rnn_embeddings[:, 1:x.shape[1], :i * length_embedding_activity],
                                             [(0, 0), (0, 1), (0, 0)], 'constant', 0)], axis=-1)

                elif self.use_present_activity:
                    # shift by one to the left
                    x = tf.concat([x, tf.pad(rnn_embeddings[:, 1:x.shape[1], :length_embedding_activity],
                                             [(0, 0), (0, 1), (0, 0)], 'constant', 0)], axis=-1)
            x = out(x)
            outputs.append(x)

        if return_state:
            return outputs, h

        return outputs

    def score(self, features, predictions):
        pass

    def detect(self, dataset):
        if isinstance(dataset, Dataset):
            features = dataset.features
        else:
            features = dataset
        predictions = self.predict(features)
        if not isinstance(predictions, list):
            predictions = [predictions]
        return AnomalyDetectionResult(scores=self.score(features, predictions), predictions=predictions)
