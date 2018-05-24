import numpy as np
import os
import pickle
from collections import Counter
from string import punctuation

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    counts = Counter(text)
    vocab = sorted(counts, key = counts.get, reverse = True)
    vocab_to_int = {word: i for i, word in enumerate(vocab)} 
    int_to_vocab = {i: word for i, word in enumerate(vocab)}
    return vocab_to_int, int_to_vocab

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    dict = {}
    dict['.'] = "||Period||"
    dict[','] = "||Comma||"
    dict['"'] = "||QuotationMark||"
    dict[';'] = "||Semicolon||"
    dict['!'] = "||Exclamationmark||"
    dict['?'] = "||Questionmark||"
    dict["("] = "||LeftParentheses||"
    dict[')'] = "||RightParentheses||"
    dict['--'] = "||Dash||"
    dict['\n'] = "||Return||"
    return dict

def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    # text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('prep.p', 'wb'))

def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))

def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('prep.p', mode='rb'))

data_dir = './data/friends/friends.txt'
text = load_data(data_dir)
preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))
sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))
