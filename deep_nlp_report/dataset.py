
from collections import Counter

from unidecode import unidecode

import numpy as np

from scipy.spatial.distance import cdist

import keras

import deep_nlp_report.utils as utils


PAD = '<PAD>'
END = '<E>'
OOV = '<OOV>'

PAD_IDX = 0
END_IDX = 1
OOV_IDX = 2


class MedianRankCallback(keras.callbacks.Callback):
    def __init__(self, test_seqs, test_positions, search_features):
        self.test_seqs = test_seqs
        self.test_positions = test_positions
        self.search_features = search_features
        
    def on_epoch_end(self, epoch, logs={}):
        test_embeddings = self.model.predict(self.test_seqs)   
        test_embeddings /= np.linalg.norm(test_embeddings, axis=-1, keepdims=True)
        all_distances = cdist(test_embeddings, self.search_features, metric='cosine')
        ranks = []
        for ds, i in zip(all_distances, self.test_positions):
            ranks.append(np.sum(ds < ds[i]))
        logs['median_dev_rank'] = np.median(ranks)
        logs['dev_rank_std'] = np.std(ranks)


class DictionaryTaskDataset:
    
    def __init__(self,
                 vocab_size, 
                 definitions_path, 
                 test_definitions_path,
                 word2vec_vec_path,
                 word2vec_word_path,
                ):

        # Read in the raw definitions (glosses) + headwords.
        raw_vocab = Counter()
        glosses = []
        headwords = []
        for line in definitions_path.open():
            line = line.strip()
            line = unidecode(line)
            line = line.lower()
            try:
                headword, gloss = line.split(' ', maxsplit=1)
                raw_vocab.update(line.split(' '))
                headwords.append(headword)
                glosses.append(gloss)
            except ValueError as e:
                assert ' ' not in line

        headwords, glosses = utils.to_array(headwords, glosses)

        # Get the pre-trained word embeddings.
        word2vec_vecs = np.load(str(word2vec_vec_path))
        word2vec_words = [w.strip() for w in word2vec_word_path.open()]
        word2vec_embeddings = {w: e for w, e in zip(word2vec_words, word2vec_vecs)}
        target_embedding_size = word2vec_vecs.shape[-1]

        # Construct a vocab from the most common words in the corpus
        # that also occur in our word embedding data
        selected_vocab = []
        for w, _ in raw_vocab.most_common():
            if w in word2vec_embeddings:
                selected_vocab.append(w)
            if len(selected_vocab) > vocab_size:
                break

        selected_vocab = [PAD, END, OOV] + list(sorted(selected_vocab))
        word_to_vocab_idx = {w: i for i, w in enumerate(selected_vocab)}

        assert word_to_vocab_idx[PAD] == PAD_IDX
        # If you're going to use Kera's helpful "mask_zeros" functionality, you must:
        assert PAD_IDX == 0
        assert word_to_vocab_idx[OOV] == OOV_IDX
        assert word_to_vocab_idx[END] == END_IDX

        glosses = [[word_to_vocab_idx.get(w, OOV_IDX) for w in g.split(' ')] + [END_IDX] for g in glosses]
        glosses = np.array([np.array(g) for g in glosses])
        
        headword_embeddings = np.zeros([len(headwords), target_embedding_size])
        valid_emb_mask = np.ones(len(headwords), dtype=bool)
        for i, hw in enumerate(headwords):
            try:
                headword_embeddings[i] = word2vec_embeddings[hw]
            except KeyError as e:
                valid_emb_mask[i] = False
        
        # Let's exclude the glosses we don't have any information for.
        headwords, glosses, headword_embeddings = \
          utils.mask_all(headwords, glosses, headword_embeddings, mask=valid_emb_mask)

        # Normalise, because we care about cosine distance.
        headword_embeddings /= np.linalg.norm(headword_embeddings, axis=-1, keepdims=True)

        # This makes the variance of each feature closer to 1, losses closer to 1.
        # because Var(X) where X is guassian in R^d is sqrt(d)
        headword_embeddings *= np.sqrt(headword_embeddings.shape[-1])

        # Construct the embedding matrix for our vocab.
        vocab_embeddings = np.zeros([len(selected_vocab), target_embedding_size])
        for i, w in enumerate(selected_vocab):
            if w not in word2vec_embeddings:
                assert w in [PAD, END, OOV]
                continue
            vocab_embeddings[i] = word2vec_embeddings[w]

        
        # Now store the stuff we care about:
        self.vocab = np.array(selected_vocab)
        self.vocab_embeddings = vocab_embeddings
        self.word_to_vocab_idx = word_to_vocab_idx
        self.glosses = glosses
        
        self.headwords = headwords
        self.headword_embeddings = headword_embeddings
        
        test_words, test_defs = [], []
        with test_definitions_path.open() as f:
            for line in f:
                word, df = line.strip().split(' ', maxsplit=1)
                test_words.append(word)
                test_defs.append(df)

        encoded_test_defs = [[word_to_vocab_idx.get(w, OOV_IDX) for w in d.split(' ')] + [END_IDX] for d in test_defs]
        encoded_test_defs = self.pad_seqs(encoded_test_defs)

        test_word_indexes = [np.searchsorted(self.vocab, tw) for tw in test_words]
        
        self.encoded_test_defs = encoded_test_defs
        self.test_word_indexes = test_word_indexes
        
        idxs = np.random.permutation(np.arange(len(headwords)))
        n_val = len(idxs) // 85
        self.n_val_examples = n_val
        self.train_idxs = idxs[:-n_val]
        self.val_idxs = idxs[-n_val:]
        
    def training_batches(self, *, batch_size):
        gs = self.glosses[self.train_idxs]
        hwes = self.headword_embeddings[self.train_idxs]
        for gloss_idxs, headword_embs in utils.batches([gs, hwes], batch_size):
            gloss_idxs = self.pad_seqs(gloss_idxs)
            yield gloss_idxs, headword_embs
            
    def val_batches(self, *, batch_size):
        gs = self.glosses[self.val_idxs]
        hwes = self.headword_embeddings[self.val_idxs]
        for gloss_idxs, headword_embs in utils.batches([gs, hwes], batch_size):
            gloss_idxs = self.pad_seqs(gloss_idxs) 
            yield gloss_idxs, headword_embs
            
    def get_median_rank_callback(self):
        return MedianRankCallback(
            self.encoded_test_defs,
            self.test_word_indexes, 
            self.vocab_embeddings
        )
    
    def pad_seqs(self, x):
        assert PAD_IDX == 0
        return keras.preprocessing.sequence.pad_sequences(x, padding='post', value=PAD_IDX)
    
    def encode_sentence(self, s):
        return [self.word_to_vocab_idx.get(w, OOV_IDX) for w in s.split(' ')] + [END_IDX]