
from types import SimpleNamespace
from pathlib import Path

from .utils import this_scripts_path


_data_root = (this_scripts_path() / '../data').resolve()


# TODO: replace with configargparse.
args = SimpleNamespace(
    
    # Data locations
    definitions_path=_data_root / 'definitions/definitions.tok', 
    test_definitions_path=_data_root / 'concept_descriptions.tok',
    word2vec_vec_path=_data_root / 'glove/glove.42B.300d.embeddings.npy',
    word2vec_word_path=_data_root / 'glove/glove.42B.300d.words.txt',
    joblib_cache_path = _data_root / 'joblib_cache',
    
    # Output
    output_loc=this_scripts_path() / '../output',
    
    # Training params
    vocab_size=100*1000,
    saves_per_epoch=5,
    max_n_epochs=100,
    batch_size=32,
    
    early_stopping_patience=15,
)
