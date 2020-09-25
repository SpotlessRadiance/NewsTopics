import numpy as np
import torch
from gensim.matutils import corpus2csc
from scipy.sparse import csc_matrix

class PaddedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, pad_idx = 0):
        self.labels = labels
        self.sent_len = max([len(text) for text in data]) + 1
        self.pad_idx = pad_idx
        self.data = self.pad_data(data)

    def pad_data(self, data):
        corpus_len = len(data)
        sent_len = max([len(text) for text in data]) + 1
        sents_corpus = np.full(fill_value=pad_idx, shape=(corpus_len, sent_len))
        for i in range(len(data)):
          text = data[i]
          sents_corpus[i][:len(text)] = text[:]
        return sents_corpus

    def __getitem__(self, idx):
        selected_data = torch.from_numpy(self.data[idx]).long()  # конвертим одну строку в нормальное представление
        selected_labels = torch.from_numpy(self.labels[idx]).long()
        return selected_data, selected_labels

    def __len__(self):
        return self.data.shape[0]


class SparseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        selected_data = torch.from_numpy(
            self.data[idx].toarray()[0]).float()  # конвертим одну строку в нормальное представление
        selected_labels = torch.from_numpy(self.labels[idx]).long()
        # selected_labels = torch.from_numpy(self.labels[idx].toarray()[0]).long()
        return selected_data, selected_labels

    def __len__(self):
        return self.data.shape[0]


def normalize_matrix(matrix, scale=True):
  matrix = matrix.multiply(1 / matrix.sum(0))  # разделить каждую строку на её длину
  if scale:
      matrix = matrix.tocsc()
      matrix -= matrix.min()
      matrix /= (matrix.max() + 1e-6)
  return matrix.tocsr()

def sparse_from_gensim(gensim_corpus):
    term_doc_mat = corpus2csc(gensim_corpus)
    term_doc_mat = term_doc_mat.transpose()
    return term_doc_mat.tocsr()