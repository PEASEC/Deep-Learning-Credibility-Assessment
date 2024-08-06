import torch
import numpy as np
from typing import Dict, List, BinaryIO, Union
from .GloveTokenizer import tokenize
from .FilePositionCache import FilePositionCache, FilePositionCacheBuilder
import weakref


class Embedding:
    _create_key = object()

    def __init__(self, create_key, embedding_size: int):
        assert (create_key == Embedding._create_key), \
            "Embeddings objects must be created using Embedding.load"
        self.rand_vector_for_oov = False
        self.__embedding_size = embedding_size

    @staticmethod
    def load(filename: str, keep_in_memory: bool = True, check_embedding_health=False) -> 'Embedding':
        """
        Loads an embedding file and returns a new Embedding instance.

        Parameters:
            filename: The path of the file

            keep_in_memory: Whether all embeddings should be load to the memory. If this flag is set
                to false, the file will be read once to index the embedding and words.
                Further requests to retrieve the embedding will result in a file read.

            check_embedding_health: If set to true, it will be checked that all embeddings have the
                same size. Increases load times.

        Returns:
            An embedding object
        """

        if keep_in_memory:
            return InMemoryEmbedding.load_from_file(filename, check_embedding_health)
        else:
            return FileBasedEmbedding.load_from_file(filename, check_embedding_health)

    @property
    def embedding_size(self):
        """
        Returns the size (dimension) of the embedding
        """
        return self.__embedding_size

    @property
    def create_random_oov_vectors(self):
        """
        Indicates the behavior if a word is not presented in the vocabulary.
            True: A unique random vector is returned
            False: A zero filled vector is returned
        """
        return self.rand_vector_for_oov

    @create_random_oov_vectors.setter
    def create_random_oov_vectors(self, value: bool):
        """
        Indicates the behavior if a word is not presented in the vocabulary.
            True: A unique random vector is returned
            False: A zero filled vector is returned
        """
        self.rand_vector_for_oov = value

    def _get_oov_vector(self):
        if self.rand_vector_for_oov:
            return torch.randn(self.__embedding_size)
        else:
            return torch.zeros(self.__embedding_size)

    def get_tweet_embeddings(self, tweet: str) -> torch.Tensor:
        """
        Creates a list of embeddings for a given tweet. The tweet is preprocessed and tokenized.
        If a word of the tweet is not in the vocabulary either a zero filled or a random tensor is returned,
        depending on the create_random_oov_vectors property.

        Parameters:
            tweet The body of the tweet

        Returns:
            A list of pytorch tensors representing the embeddings
        """
        tokens = tokenize(tweet)
        return self.get_embeddings(tokens)

    def get_embeddings(self, words: List[str]) -> torch.Tensor:
        """
        Creates a list of embeddings for a given list of words. If a word is not in the vocabulary
        either a zero filled or a random tensor is returned, depending on the create_random_oov_vectors
        property.

        Parameters:
            words A list of words

        Returns:
            A list of pytorch tensors representing the embeddings
        """
        pass

    def get_embedding(self, word: str) -> torch.Tensor:
        """
        Returns the embedding for the given word. If the given word is not in the vocabulary
        either a zero filled or a random tensor is returned, depending on the  create_random_oov_vectors
        property.

        Parameters:
            word The desired word

        Returns:
            A pytorch tensors representing the embedding
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self


class InMemoryEmbedding(Embedding):
    def __init__(self, create_key, file_name: str, embedding: Dict[str, torch.Tensor], embedding_size: int):
        super().__init__(create_key, embedding_size)
        self.file_name = file_name
        self.embedding = embedding

    @staticmethod
    def load_from_file(filename: str, check_embedding_health=False):
        result: Dict[str, torch.Tensor] = {}
        embedding_length = -1
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.split(" ")
                name = parts[0]
                vector = [float(x) for x in parts[1:]]
                result[name] = torch.tensor(vector)

                if embedding_length <= 0:
                    embedding_length = len(parts) - 1

                if check_embedding_health and embedding_length > 0:
                    assert len(parts) - 1 == embedding_length, \
                        "All embedding vectors must have the same length"

        return InMemoryEmbedding(Embedding._create_key, filename, result, embedding_length)

    def get_tweet_embeddings(self, tweet: str) -> torch.Tensor:
        tokens = tokenize(tweet)
        return self.get_embeddings(tokens)

    def get_embeddings(self, words: List[str]) -> torch.Tensor:
        if len(words) <= 0:
            return torch.empty((0, self.embedding_size))

        return torch.stack([self.get_embedding(w) for w in words])

    def get_embedding(self, word: str) -> torch.Tensor:
        if word in self.embedding:
            return self.embedding[word]
        else:
            return self._get_oov_vector()


class FileBasedEmbedding(Embedding):
    def __init__(self, create_key, file_name: str, builder: FilePositionCache[str, torch.Tensor], embedding_size: int):
        super().__init__(create_key, embedding_size)
        self.file_name = file_name
        self.file_ptr: Union[BinaryIO, None] = None
        self.embedding = builder

        self._accesses = 0

    @staticmethod
    def load_from_file(file_name: str, check_embedding_health=False):
        result: FilePositionCacheBuilder[str, torch.Tensor] = FilePositionCacheBuilder(capacity=10_000)
        embedding_length = -1
        with open(file_name, 'rb') as file:
            last_pos = 0
            binary_line = file.readline()
            while binary_line:
                name_end = binary_line.find(b" ")
                if name_end >= 0:
                    binary_name = binary_line[:name_end]
                    name = binary_name.decode()
                    result.add_item(name, last_pos)

                if embedding_length <= 0:
                    embedding_length = binary_line.count(b" ")

                if check_embedding_health and embedding_length > 0:
                    assert binary_line.count(b" ") == embedding_length, \
                        "All embedding vectors must have the same length"

                last_pos = file.tell()
                binary_line = file.readline()

        return FileBasedEmbedding(Embedding._create_key, file_name, result.build(), embedding_length)

    def __read_from_file_ptr(self, file: BinaryIO, word_list: List[str]) -> List[torch.Tensor]:
        result: List[Union[torch.Tensor, None]] = [None] * len(word_list)
        positions = {}
        for index, word in enumerate(word_list):
            if self.embedding.has_value(word):
                result[index] = self.embedding.get_value(word)
            elif self.embedding.has_pointer(word):
                positions[index] = self.embedding.get_pointer(word)
            else:
                result[index] = self._get_oov_vector()

        sorted_positions = sorted(positions.items(), key=lambda x: x[1])
        for (key, position) in sorted_positions:
            file.seek(position)
            line = file.readline().decode()
            parts = line.split(" ")
            vector = [float(x) for x in parts[1:]]
            tensor = torch.FloatTensor(vector)
            result[key] = tensor
            self.embedding.set_value(word_list[key], tensor)

        return result

    def __read_from_file(self, word_list: List[str]):
        if self.file_ptr is not None and self.file_ptr.readable():
            return self.__read_from_file_ptr(self.file_ptr, word_list)
        else:
            with open(self.file_name, 'rb') as file:
                return self.__read_from_file_ptr(file, word_list)

    def __enter__(self):
        # Do not reopen opened file
        if self.file_ptr is None:
            self.file_ptr = open(self.file_name, 'rb')

        self._accesses += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._accesses -= 1

        if self._accesses <= 0 and self.file_ptr is not None:
            self.file_ptr.close()
            self.file_ptr = None

    def get_tweet_embeddings(self, tweet: str) -> torch.Tensor:
        tokens = tokenize(tweet)
        return self.get_embeddings(tokens)

    def get_embeddings(self, words: List[str]) -> torch.Tensor:
        if len(words) <= 0:
            return torch.empty((0, self.embedding_size))

        embeddings = self.__read_from_file(words)
        return torch.stack(embeddings)

    def get_embedding(self, word: str) -> torch.Tensor:
        return self.get_embeddings([word])[0]


# Cache embeddings in an weak ref dictionary
# so multiple calls to the same embedding yield
# the same object
cache = weakref.WeakValueDictionary()


def load_embedding(filename: str, keep_in_memory: bool = True) -> Embedding:
    """
    Loads an embedding file.

    Parameters:
        filename: The path of the file

        keep_in_memory: Whether all embeddings should be load to the memory. If this flag is set
            to false, the file will be read once to index the embedding and words.
            Further requests to retrieve the embedding will result in a file read.

    Returns:
        An Embedding object
    """
    request = str(keep_in_memory) + ":" + filename
    if request in cache:
        return cache[request]
    else:
        embedding = Embedding.load(filename, keep_in_memory)
        cache[request] = embedding
        return embedding
