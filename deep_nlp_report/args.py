
from types import SimpleNamespace
from pathlib import Path


# TODO: replace with configargparse.
args = SimpleNamespace(
    
    # Data locations
    definitions_path=Path('../data/definitions/definitions.tok'), 
    test_definitions_path=Path('../data/concept_descriptions.tok'),
    word2vec_vec_path=Path('../data/glove/glove.42B.300d.embeddings.npy'),
    word2vec_word_path=Path('../data/glove/glove.42B.300d.words.txt'),
    
    # Training params
    vocab_size=100*1000,
    saves_per_epoch=200,
    max_n_epochs=100,
    batch_size=16,
)
