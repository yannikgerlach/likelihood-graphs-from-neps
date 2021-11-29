# Copyright 2020 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np
import tensorflow as tf
from april.anomalydetection.binet.attention import Attention


def binet_scores_fn(features, predictions):
    sums = [1 - np.cumsum(np.sort(p, -1), -1) for p in predictions]
    indices = [(np.argsort(p, -1) == features[:, :, i:i + 1]).argmax(-1) for i, p in enumerate(predictions)]
    scores = np.zeros(features.shape)
    for (i, j, k), f in np.ndenumerate(features):
        if f != 0 and k < len(predictions):
            scores[i, j, k] = sums[k][i, j][indices[k][i, j]]
    return scores


def detect_fn(model, features, dataset, config):
    # Get attention layers
    attention_layers = [l for l in model.layers if 'attention' in l.name]
    attention_weights = [l.output[1] for l in attention_layers]

    # Parameters
    num_predictions = len(model.outputs)
    num_attentions = len(attention_layers)
    num_attributes = dataset.num_attributes

    # Get config parameters from model architecture
    use_attributes = config.get('use_attributes', False)
    use_present_attributes = config.get('use_present_attributes', False)
    use_present_activity = config.get('use_present_activity', False)
    use_attentions = num_attentions > 0

    # Rewire outputs
    outputs = []
    if use_attentions:
        # Add attention outputs
        outputs += attention_weights
    if len(outputs) > 0:
        # We have to recompile the model to include the new outputs
        model = tf.keras.models.Model(inputs=model.inputs, outputs=model.outputs + outputs)

    # Predict
    outputs = model.predict(features)

    # Predictions must be list
    if len(model.outputs) == 1:
        outputs = [outputs]

    # Split predictions
    predictions = outputs[:num_predictions]

    # Add perfect prediction for start symbol
    for i, prediction in enumerate(predictions):
        p = np.pad(prediction[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
        p[:, 0, features[i][0, 0]] = 1
        predictions[i] = p

    # Scores
    scores = binet_scores_fn(dataset.flat_features, predictions)

    # This code can be used to load the data in batches instead of the whole dataset at once
    # scores = []
    # for i in np.arange(0, dataset.flat_features.shape[0], 500):
    #     scores.append(binet_scores_fn(dataset.flat_features[i:i + 500], [p[i:i + 500] for p in predictions]))
    # scores = np.vstack(scores)

    # Split attentions
    attentions = None
    if use_attentions:
        attentions = outputs[num_predictions:num_predictions + num_attentions]
        attentions = split_attentions(attentions, num_attributes, use_attributes, use_present_activity,
                                      use_present_attributes)

    return scores, predictions, attentions


def split_attentions(attentions, num_attributes, use_attributes, use_present_activity, use_present_attributes):
    n = num_attributes

    # Split attentions
    _attentions = []
    if not use_attributes:
        attentions += [np.zeros_like(attentions[0])] * ((n - 1) + (n - 1) * n)

    for i in range(n):
        _a = []
        for j in range(n):
            a = attentions[n * i + j]
            for __a in a:
                __a[np.triu_indices(a.shape[1], 1)] = -1
            # Add empty attentions for start symbol
            a = np.pad(a[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
            if i != 0:
                if (use_present_attributes and i != j) or (use_present_activity and j == 0):
                    a = np.pad(a[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='constant')
            _a.append(a)
        _attentions.append(_a)
    return _attentions


def binet_predictions_scores(model, dataset, features):
    # Parameters
    num_predictions = dataset.num_attributes
    # Predict
    outputs = model.predict(features)

    # Split predictions
    predictions = outputs[:num_predictions]

    # Add perfect prediction for start symbol
    for i, prediction in enumerate(predictions):
        p = np.pad(prediction[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
        p[:, 0, features[i][0, 0]] = 1
        predictions[i] = p

    # Scores
    scores = binet_scores_fn(dataset.flat_features, predictions)

    return predictions, scores


def binet_model_fn(dataset,
                   latent_dim=None,
                   use_attributes=None,
                   use_present_activity=None,
                   use_present_attributes=None,
                   encode=None,
                   decode=None,
                   use_attention=None,
                   postfix=''):
    # Validate parameters
    if latent_dim is None:
        latent_dim = min(int(dataset.max_len * 2), 64)  # clipping at 64 was not part of original paper
    if use_attributes and dataset.num_attributes == 1:
        use_attributes = False
    if use_present_attributes and dataset.num_attributes == 2:
        use_present_attributes = False
        use_present_activity = True

    if not use_attributes:
        features = dataset.features[:1]
        targets = dataset.targets[:1]
    else:
        features = dataset.features
        targets = dataset.targets

    # Build inputs (and encoders if enabled) for past events
    embeddings = []
    inputs = []
    past_outputs = []
    for feature, attr_dim, attr_key in zip(features, dataset.attribute_dims, dataset.attribute_keys):
        i = tf.keras.layers.Input(shape=(dataset.max_len,), name=f'past_{attr_key}{postfix}')
        inputs.append(i)

        voc_size = int(attr_dim + 1)  # we start at 1, hence plus 1
        emb_size = np.clip(int(voc_size / 10), 2, 16)
        embedding = tf.keras.layers.Embedding(input_dim=voc_size,
                                              output_dim=emb_size,
                                              input_length=feature.shape[1],
                                              mask_zero=True)
        embeddings.append(embedding)

        x = embedding(i)

        if encode:
            x, _ = tf.keras.layers.GRU(latent_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       name=f'past_encoder_{attr_key}{postfix}')(x)
            x = tf.keras.layers.BatchNormalization()(x)

        past_outputs.append(x)

    # Build inputs (and encoders if enabled) for present event
    present_features = []
    present_outputs = []
    if use_attributes and (use_present_activity or use_present_attributes):
        # Generate present features, by skipping the first event and adding one padding event at the end
        present_features = [np.pad(f[:, 1:], ((0, 0), (0, 1)), 'constant') for f in features]

        if use_attributes and not use_present_attributes:
            # Use only the activity features
            present_features = present_features[:1]

        for feature, embedding, attr_key in zip(present_features, embeddings, dataset.attribute_keys):
            i = tf.keras.layers.Input(shape=(dataset.max_len,), name=f'present_{attr_key}{postfix}')
            inputs.append(i)

            x = embedding(i)

            if encode:
                x = tf.keras.layers.GRU(latent_dim,
                                        return_sequences=True,
                                        name=f'present_encoder_{attr_key}{postfix}')(x)
                x = tf.keras.layers.BatchNormalization()(x)

            present_outputs.append(x)

    # Build output layers for each attribute to predict
    outputs = []
    for feature, attr_dim, attr_key in zip(features, dataset.attribute_dims, dataset.attribute_keys):
        if attr_key == 'name' or not use_attributes or (not use_present_activity and not use_present_attributes):
            x = past_outputs
        # Else predict the attribute
        else:
            x = present_outputs[:1]
            if use_present_attributes:
                for past_o, present_o, at_key in zip(past_outputs[1:], present_outputs[1:], dataset.attribute_keys[1:]):
                    if attr_key == at_key:
                        x.append(past_o)
                    else:
                        x.append(present_o)
            else:
                x += past_outputs[1:]

        if use_attention:
            attentions = []
            for _x, at_key in zip(x, dataset.attribute_keys):
                a, _ = Attention(return_sequences=True,
                                 return_coefficients=True,
                                 name=f'attention_{attr_key}/{at_key}{postfix}')(_x)
                attentions.append(a)
            x = attentions
        if len(x) > 1:
            x = tf.keras.layers.concatenate(x)
        else:
            x = x[0]

        if decode:
            x = tf.keras.layers.GRU(latent_dim,
                                    return_sequences=True,
                                    name=f'decoder_{attr_key}{postfix}')(x)
            x = tf.keras.layers.BatchNormalization()(x)

        o = tf.keras.layers.Dense(int(attr_dim + 1), activation='softmax', name=f'out_{attr_key}{postfix}')(x)
        outputs.append(o)

    # Combine features and build model
    features = features + present_features
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy'
    )

    return model, features, targets
