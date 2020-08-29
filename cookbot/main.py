import os
import re
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .encoder import Encoder
from .decoder import Decoder

BATCH_SIZE = 29
EMBEDDING_DIMS = 256
RNN_UNITS = 1024
DENSE_UNITS = 1024

def preprocess_sequences(data, tokenizer):
    tokenizer.fit_on_texts(data)

    #convert text to sequence
    seq = tokenizer.texts_to_sequences(data)
    seq = pad_sequences(seq, padding='post')
    return len(seq), seq

def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max( len(t) for t in tensor)

#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
    return [tf.zeros((BATCH_SIZE, RNN_UNITS)), tf.zeros((BATCH_SIZE, RNN_UNITS))]


def run():
    open('cookbot/log.txt', 'w').close() # clear log before writing to it
    log = open('cookbot/log.txt', 'a')

    with open('cookbot/recipelinks.json') as recipe_links:
        recipes = json.load(recipe_links)
        BATCH_SIZE = len(recipes['recipes'])

        tokenizer = Tokenizer(filters='!#$%&=+,-/:;=?@^_`{|}~\t') # default but removed < and >

        inputs = list()
        labels = list()

        for recipe_obj in recipes['recipes']:
            try:
                recipe_response = requests.get(recipe_obj['link'])
                recipe_soup = BeautifulSoup(recipe_response.text, 'lxml')

                recipe = re.sub(
                    r'\n\s*\n', '\n\n', recipe_soup.body.get_text()) # compress multiple blank lines

                # recipe = '<start> ' + recipe + ' <end>'

                recipe_output = ''.join([item.join(' ,') for item in recipe_obj['desired_output']])
                # recipe_output = ' <start> ' + recipe_output + ' <end>'

                # recipe_pair = [recipe, recipe_output]

                # recipe_preprocess = preprocess_pairs([recipe_pair], tokenizer)
                # print(recipe_preprocess)

                # full_recipe = list()
                # trimmed_recipe = list()
                # for data in recipe_preprocess:
                #     full_recipe.append(data[0]), trimmed_recipe.append(data[1])
                inputs.append(recipe)
                labels.append(recipe_output)
            except Exception:
                continue

        labels = ['<start> ' + output + ' <end>' for output in labels]

        log.writelines([(label + '\n') for label in labels])
        size, input_encoded = preprocess_sequences(inputs, tokenizer)

        label_encoded = preprocess_sequences(labels, tokenizer)[1]

        # X_train,  X_test, Y_train, Y_test = train_test_split(input_encoded, label_encoded, test_size=0.2)
        BUFFER_SIZE = len(input_encoded)
        steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
        print('size', BUFFER_SIZE)

        Tx = max_len(input_encoded)
        Ty = max_len(label_encoded)

        vocab_size = len(tokenizer.word_index) + 1
        dataset = tf.data.Dataset.from_tensor_slices((input_encoded, label_encoded)).batch(BATCH_SIZE, drop_remainder=True)

        encoder = Encoder(vocab_size, EMBEDDING_DIMS, RNN_UNITS)
        decoder = Decoder(vocab_size, EMBEDDING_DIMS, RNN_UNITS)
        optimizer = tf.keras.optimizers.Adam()

        checkpoint_dir = '../training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            encoder=encoder,
            decoder=decoder
        )
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        epochs = 5
        for e in range(epochs):
            if e > 0 and e % 10 == 0:
                print(checkpoint_prefix)
                checkpoint.save(file_prefix=checkpoint_prefix)

            en_initial_states = encoder.init_states(BATCH_SIZE)

            for batch, (source_seq, target_seq) in enumerate(dataset):
                loss = train_step(
                    encoder, decoder, optimizer,
                    source_seq, target_seq,
                    target_seq, en_initial_states
                )

            print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

        # preprocess raw data to predict
        link = "https://minimalistbaker.com/best-vegan-matcha-latte/"
        recipe_response = requests.get(link)
        recipe_soup = BeautifulSoup(recipe_response.text, 'lxml')

        recipe = re.sub(
            r'\n\s*\n', '\n', recipe_soup.body.get_text()) # compress multiple blank lines
        # recipe = '<start> ' + recipe

        predict(inputs[1], tokenizer, encoder, decoder)

def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

def train_step(encoder, decoder, optimizer, source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        # We need to create a loop to iterate through the target sequences
        for i in range(target_seq_out.shape[1]):
            # Input to the decoder must have shape of (batch_size, length)
            # so we need to expand one dimension
            decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
            logit, de_state_h, de_state_c, _ = decoder(
                decoder_in,
                (de_state_h, de_state_c),
                en_outputs[0]
            )

            # The loss is now accumulated through the whole batch
            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]


def predict(raw_data, tokenizer, encoder, decoder):
    if raw_data is None:
        raw_data = raw_data[np.random.choice(len(raw_data))]
    # print(raw_data)
    test_source_seq = tokenizer.texts_to_sequences([raw_data])
    # print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(
            de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        out_words.append(tokenizer.index_word[de_input.numpy()[0][0]])

        alignments.append(alignment.numpy())

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    return np.array(alignments), raw_data.split(' '), out_words
