# A standard transformer architecture with input and evaluation pipelines for protein primary structure

# Importing modules
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
import time
import logo_maker
import Write_Results


# Creates a tensorflow dataset from a local CSV file
def create_tf_dataset(csv_dataset_location):
    ds = pd.read_csv(csv_dataset_location)
    ds_labels_raw = ds.copy()
    ds_features_raw = ds_labels_raw.pop("ancestral_sequence")
    number_of_records = len(ds["descendant_sequence"])

    ds_labels = pd.DataFrame(columns=["descendant_sequence"])
    for i in range(0, number_of_records):
        ds_labels.loc[i] = (ds_labels_raw["descendant_sequence"])[i]

    ds_features = pd.DataFrame(columns=["ancestral_sequence"])
    for i in range(0, number_of_records):
        ds_features.loc[i] = ds_features_raw[i]

    tf_dataset = tf.data.Dataset.from_tensor_slices((ds_features, ds_labels))
    print("Total records from source:", number_of_records)
    tf_dataset = tf_dataset.shuffle(buffer_size=number_of_records, reshuffle_each_iteration=True)
    return [tf_dataset, number_of_records]

# Creates training, validation, and test datasets from source dataset
def create_datasets(dataset, number_of_records):
    dataset = dataset.shuffle(buffer_size=number_of_records, reshuffle_each_iteration=False)
    train_ds_size = int(0.8 * number_of_records)
    test_ds_size = int(0.1 * number_of_records)
    val_ds_size = int(0.1 * number_of_records)

    train_ds = dataset.take(train_ds_size)
    test_and_val_ds = dataset.skip(train_ds_size)
    val_ds = test_and_val_ds.skip(test_ds_size)
    test_ds = test_and_val_ds.take(test_ds_size)
    print("Training dataset size: ", train_ds)
    print("Validation dataset size: ", val_ds)
    print("Test dataset size: ", test_ds)

    return train_ds, train_ds_size, val_ds, val_ds_size, test_ds, test_ds_size


# Creating vectorization layer
class Create_Vectorization_Layer:
    def __init__(self, tf_dataset, vocab_size=22, sequence_length=786):
        self.tf_dataset = tf_dataset
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.vectorize = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=self.sequence_length)

        self.vectorize.adapt(["l", "v", "g", "s", "t", "a", "i", "n", "e", "k", "d", "p", "r", "f", "q", "y", "c", "h",
                              "w", "m"])
        self.vocabulary = self.vectorize.get_vocabulary()

    def vectorize_tensor_pair(self, feature, label):
        vectorized_feature = np.insert(self.vectorize(feature).numpy(), 0, self.vocab_size)
        vectorized_feature = np.insert(vectorized_feature, len(vectorized_feature), self.vocab_size + 1)
        vectorized_label = np.insert(self.vectorize(label).numpy(), 0, self.vocab_size)
        vectorized_label = np.insert(vectorized_label, len(vectorized_label), self.vocab_size + 1)

        return vectorized_feature, vectorized_label

    def wrap_vectorize_tensor_pair(self, feature, label):
        vectorized_features, vectorized_labels = tf.py_function(self.vectorize_tensor_pair, [feature, label],
                                                                [tf.int64, tf.int64])
        vectorized_features.set_shape([self.sequence_length + 2])
        vectorized_labels.set_shape([self.sequence_length + 2])

        return vectorized_features, vectorized_labels

    def vectorize_dataset(self, dataset):
        return dataset.map(self.wrap_vectorize_tensor_pair)


# Batch and optimise a dataset for improved training performance
def batch_and_optimise(dataset, size, batch_size=10):
    dataset.shuffle(size)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# calculating angles for positional encoding
def get_angles(pos, i, embedding_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
    return pos * angle_rates


# Applying positional encoding
def positional_encoding(pos, embedding_dim):
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis],
                            np.arange(embedding_dim)[np.newaxis, :],
                            embedding_dim)

    # applying sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # applying cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# Creating padding mask for a batch of sequences
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]


# Creating look-ahead mask
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    return mask


# Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# Multihead attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        assert embedding_dim % self.num_heads == 0

        self.depth = embedding_dim // self.num_heads

        self.wq = tf.keras.layers.Dense(embedding_dim)
        self.wk = tf.keras.layers.Dense(embedding_dim)
        self.wv = tf.keras.layers.Dense(embedding_dim)

        self.dense = tf.keras.layers.Dense(embedding_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, embedding_dim)
        k = self.wk(k)  # (batch_size, seq_len, embedding_dim)
        v = self.wv(v)  # (batch_size, seq_len, embedding_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.embedding_dim))  # (batch_size, seq_len_q, embedding_dim)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, embedding_dim)

        return output, attention_weights


# Pointwise feed forward neural network
def point_wise_feed_forward_network(embedding_dim, ffn_width):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(ffn_width, activation='relu'),  # (batch_size, seq_len, ffn_width)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, embedding_dim)
    ])


# Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ffn_width, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_dim, ffn_width)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, embedding_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embedding_dim)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embedding_dim)

        return out2


# Decoder layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ffn_width, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(embedding_dim, num_heads)
        self.mha2 = MultiHeadAttention(embedding_dim, num_heads)

        self.ffn = point_wise_feed_forward_network(embedding_dim, ffn_width)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, embedding_dim)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, embedding_dim)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, embedding_dim)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, embedding_dim)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, embedding_dim)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embedding_dim)

        return out3, attn_weights_block1, attn_weights_block2


# Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, encoder_layers, embedding_dim, num_heads, ffn_width, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.embedding_dim)

        self.enc_layers = [EncoderLayer(embedding_dim, num_heads, ffn_width, rate)
                           for _ in range(encoder_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.encoder_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, embedding_dim)


# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, decoder_layers, embedding_dim, num_heads, ffn_width, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.decoder_layers = decoder_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dim)

        self.dec_layers = [DecoderLayer(embedding_dim, num_heads, ffn_width, rate)
                           for _ in range(decoder_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.decoder_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, embedding_dim)
        return x, attention_weights


# Transformer
class Transformer(tf.keras.Model):
    def __init__(self, encoder_layers, decoder_layers, embedding_dim, num_heads, ffn_width, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(encoder_layers, embedding_dim, num_heads, ffn_width,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(decoder_layers, embedding_dim, num_heads, ffn_width,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, embedding_dim)


        # dec_output.shape == (batch_size, tar_seq_len, embedding_dim)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, embedding_dim)

        return final_output, attention_weights


# Custom learning rate schedule for Adam optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding_dim = tf.cast(self.embedding_dim, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embedding_dim) * tf.math.minimum(arg1, arg2)


# Applying padding mask to loss calculation
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# Applying padding mask to accuracy calculation
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


# Applying padding and lookahead masks
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# Creating training steps
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


# Validation and testing
def validation_and_testing(inp, tar, type):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(inp, tar_inp,
                                 False,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

    if type == "val":
        val_loss(loss)
        val_accuracy(accuracy_function(tar_real, predictions))
    elif type == "test":
        test_loss(loss)
        test_accuracy(accuracy_function(tar_real, predictions))


# Retrieving descendant DNA prediction for a single ancestral DNA sequence
def retrieve_descendant_prediction(ancestor_sequence):
    ancestor_sequence = tf.convert_to_tensor(ancestor_sequence)
    ancestor_sequence = tf.expand_dims(ancestor_sequence, 0)
    ancestor_sequence = np.insert(vectorization_layer.vectorize(ancestor_sequence).numpy(), 0,
                                 vectorization_layer.vocab_size)
    ancestor_sequence = np.insert(ancestor_sequence, len(ancestor_sequence), vectorization_layer.vocab_size + 1)

    encoder_input = tf.expand_dims(ancestor_sequence, 0)

    decoder_input = [vectorization_layer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    nt_probabilities = {}
    for i in range(vectorization_layer.sequence_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        probablities = tf.nn.softmax(predictions, axis=-1)
        probablities = probablities.numpy()
        max_prob = np.amax(probablities)
        nt_probabilities[i] = [max_prob]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == vectorization_layer.vocab_size + 1:    # vocab size + 1
            return tf.squeeze(output, axis=0), attention_weights, nt_probabilities

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights, nt_probabilities


def predict_descendant(ancestor_sequence, epoch):
    result, attention_weights, nt_probabilities = retrieve_descendant_prediction(ancestor_sequence)
    vocabulary = vectorization_layer.vocabulary
    print(result)

    result = result.numpy()
    predicted_descendant_design = ""
    seq_index = 0
    for i in result:
        if i < vectorization_layer.vocab_size:
            nt_probabilities[seq_index].append(vocabulary[i])
            predicted_descendant_design += vocabulary[i]
            seq_index += 1

    ancestor_sequence = ancestor_sequence.replace(" ", "")
    plot_all_heads(ancestor_sequence, predicted_descendant_design, attention_weights, epoch)
    print('Ancestor sequence: {}'.format(ancestor_sequence))
    print('Predicted descendant sequence: ', predicted_descendant_design.upper())
    written_results["predicted_descendant_design"].append(predicted_descendant_design)
    logo_maker.create_probability_logo(nt_probabilities, vectorization_layer.vocabulary, write_directory, epoch)


# Plot training and validation loss
def plot_loss(train_history, val_history):
    train_loss_list = []
    val_loss_list = []
    for key in train_history.keys():
        train_loss_list.append(train_history[key][0])
    for key in val_history.keys():
        val_loss_list.append(val_history[key][0])

    plt.figure()
    plt.plot(train_loss_list, label='loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.ylim([0, 5])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    Write_Results.write_image(plt, "loss_graph", write_directory, max(EPOCHS))


# Plot training and validation accuracy
def plot_accuracy(train_history, val_history):
    train_acc_list = []
    val_acc_list = []
    for key in train_history.keys():
        train_acc_list.append(train_history[key][1])
    for key in val_history.keys():
        val_acc_list.append(val_history[key][1])

    plt.figure()
    plt.plot(train_acc_list, label='accuracy')
    plt.plot(val_acc_list, label='val_accuracy')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()
    Write_Results.write_image(plt, "acc_graph", write_directory, max(EPOCHS))


# Plot heat map for a single attention head
def plot_attention_head(ancestor_sequence, predicted_descendant_sequence, head, head_identity, epoch, layer_number):
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    ax.matshow(head)

    while len(ancestor_sequence) < vectorization_layer.sequence_length:
        ancestor_sequence = ancestor_sequence + " "
    ancestor_sequence = ["<Start>"] + [nt.upper() for nt in ancestor_sequence] + ["<End>"]
    predicted_descendant_sequence = [nt.upper() for nt in predicted_descendant_sequence]

    ax.set_xticks(range(len(ancestor_sequence)))
    ax.set_yticks(range(len(predicted_descendant_sequence)))

    x_labels = []
    for pos, nt in enumerate(ancestor_sequence):
        if nt == "<Start>" or nt == "<End>":
            x_labels.append(nt)
            continue
        elif pos % 10 == 0:
            x_labels.append(nt + " #" + str(pos))
        else:
            x_labels.append(nt)

    y_labels = []
    for pos, nt in enumerate(predicted_descendant_sequence):
        pos += 1
        if nt == "<Start>" or nt == "<End>":
            y_labels.append(nt)
            continue
        elif pos % 10 == 0:
            y_labels.append(("#" + str(pos)) + " " + nt)
        else:
            y_labels.append(nt)

    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Head {}'.format(head_identity))

    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False, )

    Write_Results.write_image(plt, "attention_head", write_directory, epoch, layer_number)
    plt.close(fig)


# Plot heat map for all attention heads in a given multi-head attention layer
def plot_all_heads(ancestor_sequence, predicted_descendant_sequence, attention_dict, epoch):
    if all_heatmaps:
        for layer_number in range(1, decoder_layers + 1):
            attention_heads = tf.squeeze(attention_dict['decoder_layer' + str(layer_number) + '_block2'], 0)
            for h, head in enumerate(attention_heads):
                head_identity = h + 1
                plot_attention_head(ancestor_sequence, predicted_descendant_sequence, head, head_identity, epoch,
                                    str(layer_number))
    else:
        attention_heads = tf.squeeze(
            attention_dict['decoder_layer' + str(decoder_layers) + '_block2'], 0)

        for h, head in enumerate(attention_heads):
            head_identity = h + 1
            plot_attention_head(ancestor_sequence, predicted_descendant_sequence, head, head_identity, epoch, str(decoder_layers))


# MODEL PARAMETERS AND DATA INPUT
dataset_folder_directory = ""      # Directory for the folder that contains your prepared dataset(s)
dataset_identity = ""   # The name of the dataset file (do NOT include .csv file extension)
encoder_layers = 1
decoder_layers = 1
embedding_dim = 10
ffn_width = 20
num_heads = 2
dropout_rate = 0.8
EPOCHS = [1, 2, 5]
all_heatmaps = False    # If you want to save images of all attention heatmaps, set this to true.

# ENTER THE ANCESTRAL SEQUENCE YOU WANT TO PREDICT ON, AS A CONTINUOUS STRING (NO SPACES), HERE:
ancestor_sequence = ""

schedule_identity = "0"
write_directory = Write_Results.assign_local_directory(schedule_identity)


# Creating a dictionary, for written results
written_results = {"predicted_descendant_design": []}

source_dataset, dataset_size = create_tf_dataset(dataset_folder_directory + "\\" + dataset_identity + ".csv")
train_ds, train_ds_size, val_ds, val_ds_size, test_ds, test_ds_size = create_datasets(source_dataset, dataset_size)

vectorization_layer = Create_Vectorization_Layer(train_ds)

train_ds = batch_and_optimise(vectorization_layer.vectorize_dataset(train_ds), size=train_ds_size)
val_ds = batch_and_optimise(vectorization_layer.vectorize_dataset(val_ds), size=val_ds_size)
test_ds = batch_and_optimise(vectorization_layer.vectorize_dataset(test_ds), size=test_ds_size)

# Loss and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
train_history = {}  # Epoch: loss, accuracy
val_history = {}  # Epoch: loss, accuracy
written_results["test_loss"] = []
written_results["test_accuracy"] = []

# Setting hyperparameters
input_vocab_size = vectorization_layer.vocab_size + 2  # Adding two for start and end tokens
target_vocab_size = vectorization_layer.vocab_size + 2
input_seq_length = vectorization_layer.sequence_length + 2  # Adding two for start and end tokens
output_seq_length = vectorization_layer.sequence_length + 2

ancestor_sequence_spaced = ""
for i in ancestor_sequence:
    ancestor_sequence_spaced += " " + i
ancestor_sequence_spaced = ancestor_sequence_spaced.strip()

# Setting learning rate and optimiser
learning_rate = CustomSchedule(embedding_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# Creating transformer
transformer = Transformer(encoder_layers, decoder_layers, embedding_dim, num_heads, ffn_width,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_seq_length,
                          pe_target=output_seq_length,
                          rate=dropout_rate)

# Creating checkpoint path and checkpoint manager

checkpoint_path = "checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

temp_learning_rate_schedule = CustomSchedule(embedding_dim)

# Training, validation, and testing
for epoch in range(max(EPOCHS)):
    start = time.time()

    # Training
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_ds):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    # Validation
    val_loss.reset_states()
    val_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(val_ds):
        validation_and_testing(inp, tar, "val")

    print('Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}'.format(
        epoch + 1, val_loss.result(), val_accuracy.result()))

    # Testing and predicting
    if epoch + 1 in EPOCHS:
        test_loss.reset_states()
        test_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(test_ds):
            validation_and_testing(inp, tar, "test")

        print('Test Loss {:.4f} Test Accuracy {:.4f}'.format(test_loss.result(), test_accuracy.result()))
        written_results["test_loss"].append(str(test_loss.result().numpy()))
        written_results["test_accuracy"].append(str(test_accuracy.result().numpy()))
        predict_descendant(ancestor_sequence_spaced, epoch + 1)

    train_history[epoch + 1] = [train_loss.result().numpy(), train_accuracy.result().numpy()]
    val_history[epoch + 1] = [val_loss.result().numpy(), val_accuracy.result().numpy()]

plot_loss(train_history, val_history)
plot_accuracy(train_history, val_history)

# Storing information for writing to file
written_results["encoder_layers"] = str(encoder_layers)
written_results["decoder_layers"] = str(decoder_layers)
written_results["embedding_dim"] = str(embedding_dim)
written_results["ffn_width"] = str(ffn_width)
written_results["num_heads"] = str(num_heads)
written_results["input_vocab_size"] = str(input_vocab_size)
written_results["target_vocab_size"] = str(target_vocab_size)
written_results["dropout_rate"] = str(dropout_rate)
written_results["Epochs"] = EPOCHS
written_results["input_seq_length"] = str(input_seq_length)
written_results["output_seq_length"] = str(output_seq_length)
written_results["ancestor_sequence"] = ancestor_sequence
written_results["dataset_identity"] = dataset_identity
written_results["schedule identity"] = schedule_identity
written_results["train_ds_size"] = str(train_ds_size)
written_results["val_ds_size"] = str(val_ds_size)
written_results["test_ds_size"] = str(test_ds_size)

Write_Results.write_results(write_directory, written_results)
plt.close('all')
