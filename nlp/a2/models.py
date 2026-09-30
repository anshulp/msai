# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class CharacterLSTM(nn.Module):
    """Produces an extra word embedding from characters to help with misspellings."""

    def __init__(self, embedding_dim: int, hidden_size: int = 25):
        super().__init__()
        alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        self.char_to_index = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.char_to_index["<pad>"] = 0
        self.char_embedding = nn.Embedding(len(self.char_to_index), hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, embedding_dim)

    def encode_word(self, word: str) -> torch.Tensor:
        if len(word) == 0:
            word = "<pad>"
        char_ids = torch.tensor([self.char_to_index.get(ch, self.char_to_index["<pad>"]) for ch in word], dtype=torch.long).unsqueeze(0)
        embedded = self.char_embedding(char_ids)
        _, (hidden, _) = self.lstm(embedded)
        return self.output(hidden[-1])

    def encode_batch(self, words_list: List[List[str]]) -> torch.Tensor:
        batch_vectors = []
        for words in words_list:
            sent_vectors = []
            for word in words:
                sent_vectors.append(self.encode_word(word))
            if len(sent_vectors) == 0:
                sent_vectors.append(torch.zeros(1, self.output.out_features, device=self.output.weight.device))
            batch_vectors.append(torch.cat(sent_vectors, dim=0))
        return batch_vectors


class DeepAveragingNetwork(nn.Module):
    """Deep averaging network over word embeddings for sentiment classification."""

    def __init__(self, embedding_layer: nn.Embedding, hidden_size: int, num_classes: int = 2, char_lstm: CharacterLSTM = None):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.hidden_size = hidden_size
        self.char_lstm = char_lstm
        self.fc1 = nn.Linear(embedding_layer.embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, token_ids: torch.Tensor, batch_words: List[List[str]] = None, use_char_lstm: bool = False) -> torch.Tensor:
        """
        token_ids: shape (batch_size, seq_len)
        returns: shape (batch_size, num_classes) log-probability distribution
        """
        embedded = self.embedding_layer(token_ids)  # (batch, seq_len, emb_dim)
        if use_char_lstm and self.char_lstm is not None and batch_words is not None:
            char_emb = []
            for sent_words in batch_words:
                sent_vecs = []
                for word in sent_words:
                    sent_vecs.append(self.char_lstm.encode_word(word))
                if len(sent_vecs) == 0:
                    sent_vecs.append(torch.zeros(1, embedded.shape[-1], device=embedded.device))
                char_emb.append(torch.cat(sent_vecs, dim=0))
            for i in range(len(char_emb)):
                pad_len = max(0, embedded.shape[1] - char_emb[i].shape[0])
                if pad_len > 0:
                    char_emb[i] = torch.cat([char_emb[i], torch.zeros(pad_len, embedded.shape[-1], device=embedded.device)], dim=0)
                char_emb[i] = char_emb[i][:embedded.shape[1]]
            char_emb = torch.stack(char_emb)
            embedded = embedded + char_emb
        averaged = embedded.mean(dim=1)  # (batch, emb_dim)
        hidden = self.relu(self.fc1(averaged))
        logits = self.fc2(hidden)
        return self.log_softmax(logits)


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Wraps a deep averaging network for sentiment classification.
    """

    def __init__(self, word_embeddings: WordEmbeddings, hidden_size: int = 100, frozen_embeddings: bool = False, use_char_lstm: bool = True):
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=frozen_embeddings)
        self.char_lstm = CharacterLSTM(word_embeddings.get_embedding_length(), hidden_size=25) if use_char_lstm else None
        self.network = DeepAveragingNetwork(self.embedding_layer, hidden_size, num_classes=2, char_lstm=self.char_lstm)
        self._typo_normalization_cache = {}
        self._typo_candidates_by_length = {}
        for candidate in self.word_indexer.objs_to_ints:
            if candidate not in ("PAD", "UNK"):
                self._typo_candidates_by_length.setdefault(len(candidate), []).append(candidate)

    def _edit_distance(self, left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)

        prev = list(range(len(right) + 1))
        for i, ch_left in enumerate(left, 1):
            curr = [i]
            for j, ch_right in enumerate(right, 1):
                insertion = curr[j - 1] + 1
                deletion = prev[j] + 1
                substitution = prev[j - 1] + (ch_left != ch_right)
                curr.append(min(insertion, deletion, substitution))
            if min(curr) > 2:
                return 3
            prev = curr
        return prev[-1]

    def _normalize_word_for_typo(self, word: str) -> str:
        if word == "":
            return "UNK"

        normalized = word.lower()
        if self.word_indexer.index_of(normalized) != -1:
            return normalized
        cached = self._typo_normalization_cache.get(normalized)
        if cached is not None:
            return cached

        best_candidates = []
        for candidate_length in range(max(1, len(normalized) - 2), len(normalized) + 3):
            for candidate in self._typo_candidates_by_length.get(candidate_length, []):
                if len(normalized) > 4 and candidate[0] != normalized[0]:
                    continue

                dist = self._edit_distance(normalized, candidate)
                if dist <= 2:
                    best_candidates.append((dist, candidate))

        if not best_candidates:
            self._typo_normalization_cache[normalized] = "UNK"
            return "UNK"

        best_candidates.sort(key=lambda pair: (pair[0], len(pair[1])))
        correction = best_candidates[0][1]
        self._typo_normalization_cache[normalized] = correction
        return correction

    def _tokenize(self, ex_words: List[str], has_typos: bool = False) -> torch.Tensor:
        if len(ex_words) == 0:
            ex_words = ["UNK"]
        indices = []
        for word in ex_words:
            lookup_word = self._normalize_word_for_typo(word) if has_typos else word
            idx = self.word_indexer.index_of(lookup_word)
            if idx == -1:
                idx = self.word_indexer.index_of("UNK")
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)

    def _batch_tokenize(self, batch_ex_words: List[List[str]], has_typos: bool = False) -> torch.Tensor:
        max_len = max(len(ex_words) if len(ex_words) > 0 else 1 for ex_words in batch_ex_words)
        pad_idx = self.word_indexer.index_of("PAD")
        batch = []
        for ex_words in batch_ex_words:
            seq = ex_words if len(ex_words) > 0 else ["UNK"]
            ids = []
            for word in seq:
                lookup_word = self._normalize_word_for_typo(word) if has_typos else word
                idx = self.word_indexer.index_of(lookup_word)
                if idx == -1:
                    idx = self.word_indexer.index_of("UNK")
                ids.append(idx)
            ids += [pad_idx] * (max_len - len(ids))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        self.network.eval()
        with torch.no_grad():
            token_ids = self._tokenize(ex_words, has_typos=has_typos).unsqueeze(0)
            log_probs = self.network(token_ids, batch_words=[ex_words], use_char_lstm=self.char_lstm is not None)
            return int(torch.argmax(log_probs, dim=1).item())

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        if len(all_ex_words) == 0:
            return []
        self.network.eval()
        with torch.no_grad():
            token_ids = self._batch_tokenize(all_ex_words, has_typos=has_typos)
            log_probs = self.network(token_ids, batch_words=all_ex_words, use_char_lstm=self.char_lstm is not None)
            return [int(pred) for pred in torch.argmax(log_probs, dim=1).tolist()]


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    model = NeuralSentimentClassifier(
        word_embeddings,
        hidden_size=getattr(args, 'hidden_size', 100),
        frozen_embeddings=False,
        use_char_lstm=train_model_for_typo_setting,
    )
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.network.parameters(), lr=getattr(args, 'lr', 0.001))
    batch_size = getattr(args, 'batch_size', 1)

    for epoch in range(getattr(args, 'num_epochs', 10)):
        random.shuffle(train_exs)
        model.network.train()
        for start in range(0, len(train_exs), batch_size):
            batch = train_exs[start:start + batch_size]
            batch_token_ids = model._batch_tokenize([ex.words for ex in batch])
            batch_words = [ex.words for ex in batch]
            targets = torch.tensor([ex.label for ex in batch], dtype=torch.long)

            model.network.zero_grad()
            log_probs = model.network(batch_token_ids, batch_words=batch_words, use_char_lstm=train_model_for_typo_setting)
            loss = criterion(log_probs, targets)
            #print("Epoch %d, Batch %d: Loss = %.4f" % (epoch + 1, start // batch_size + 1, loss.item()))
            loss.backward()
            optimizer.step()

    model.network.eval()
    #done
    return model

