from gensim.models.word2vec import LineSentence
from gensim.models import Phrases
from gensim.test.utils import common_texts
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
path = get_tmpfile("word2vec.model")
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=1)

vector = model.wv['computer']  # numpy vector of a word

path = get_tmpfile("wordvectors.kv")
model.wv.save(path)
wv = KeyedVectors.load("model.wv", mmap='r')
vector = wv['computer']  # numpy vector of a word

wv_from_text = KeyedVectors.load_word2vec_format(
    datapath('word2vec_pre_kv_c'), binary=False)  # C text format
wv_from_bin = KeyedVectors.load_word2vec_format(
    datapath("euclidean_vectors.bin"), binary=True)  # C binary format

word_vectors = model.wv
del model

bigram_transformer = Phrases(common_texts)
model = Word2Vec(bigram_transformer[common_texts], min_count=1)

sentences = LineSentence(datapath('lee_background.cor'))
for sentence in sentences:
    pass

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(min_count=1)
model.build_vocab(sentences)  # prepare the model vocabulary
model.train(sentences, total_examples=model.corpus_count,
            epochs=model.iter)  # train word vectors
