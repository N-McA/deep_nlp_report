
from .args import args
from .dataset import DictionaryTaskDataset

from functools import lru_cache

import keras


@lru_cache(1)
def get_dataset():
    return DictionaryTaskDataset(
        vocab_size=args.vocab_size, 
        definitions_path=args.definitions_path, 
        test_definitions_path=args.test_definitions_path,
        word2vec_vec_path=args.word2vec_vec_path,
        word2vec_word_path=args.word2vec_word_path,
    )


def test_model(model, model_name):
    
    dictionary_dataset = get_dataset()
    ds = dictionary_dataset
    
    opt = keras.optimizers.Adam(amsgrad=True)
    model.compile(opt, loss='mse')
    
    model.fit_generator(
        dictionary_dataset.training_batches(batch_size=args.batch_size),
        steps_per_epoch=len(dictionary_dataset.glosses) // (args.saves_per_epoch * args.batch_size),
        epochs=args.max_n_epochs,
        callbacks=[
            dictionary_dataset.get_median_rank_callback(),
            keras.callbacks.CSVLogger('{}_training.log'.format(model_name)),
            keras.callbacks.ModelCheckpoint('weights/' + model_name + '.{epoch:02d}.hdf5')
        ]
    )
    