import re
import numpy as np
import pymorphy2
import gensim
import nltk
from gensim.models import TfidfModel

def corpus_tokens(corpus):
  '''
  :param corpus: array with texts
  :return: numpy 2-dimensional array with lemmatized words
  '''
  stopwords = nltk.corpus.stopwords.words('russian')
  WORD_RE = re.compile(r'[а-я]+-{,1}[а-я]*')
  morph = pymorphy2.MorphAnalyzer()
  tokens = []
  for text in corpus:
    text = text.lower()
    words = re.findall(WORD_RE, text)
    tokens.append(np.array([morph.normal_forms(word)[0] for word in words if word not in stopwords]))
  return np.array(tokens)

def create_bigrams(tokenized_corpus):
  '''
  :param tokenized_corpus: 2-dimensional array with tokens from one text in each row
  :return: 2-dimensional array with tokens, often found together are combined into bigrams
  '''

  bigram = gensim.models.Phrases(tokenized_corpus, min_count=5, threshold=50)
  bigram_mod = gensim.models.phrases.Phraser(bigram)
  for i in range(len(tokenized_corpus)):
    tokenized_corpus[i] = bigram_mod[tokenized_corpus[i]]
  return tokenized_corpus


def corpus_to_bow(tokenized_corpus):
  '''
  :param tokenized_corpus: 2-dimensional array with tokens from one text in each row
  :return:
     id2word: gensim.corpora.Dictionary
     corpus: array with texts in BoW-format (list of (token_id, token_count) tuples in each row)
  '''
  id2word = gensim.corpora.Dictionary(tokenized_corpus)
  id2word.filter_n_most_frequent(100)
  corpus = [id2word.doc2bow(text) for text in tokenized_corpus]
  return id2word, corpus

def corpus_to_idx(tokenized_corpus, id2word = None):
  '''
  :param tokenized_corpus: 2-dimensional array with tokens from one text in each row
  :return:
  '''
  if id2word == None:
      id2word = gensim.corpora.Dictionary(tokenized_corpus)
      id2word.filter_n_most_frequent(100)
  unk_index = len(id2word)
  corpus = np.array([np.array(id2word.doc2idx(text, unk_index)) for text in tokenized_corpus])
  return corpus, id2word


def filter_tf_idf(tokenized_corpus, corpus = None, id2word = None):
  '''
  Filter words with low tf-idf
  :param tokenized_corpus: 2-dimensional array with tokens from one text in each row
         id2word: gensim.corpora.Dictionary
         corpus: array with texts in BoW-format (list of (token_id, token_count) tuples in each row)
  :return:
         id2word: renewed gensim.corpora.Dictionary
         corpus: renewed corpus
    '''
  if corpus == None or id2word == None:
    id2word, corpus = corpus_to_bow(tokenized_corpus)
  model = TfidfModel(corpus, id2word=id2word)
  low_value = 0.02
  low_value_words = []
  for bow in corpus:
    low_value_words += [id for id, value in model[bow] if value < low_value]
  id2word.filter_tokens(bad_ids=low_value_words)
  corpus = [id2word.doc2bow(text) for text in tokenized_corpus]
  return id2word, corpus