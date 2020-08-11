import numpy as np
from typing import List, Callable, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('punkt')
nltk.download('wordnet')


def find_similar_document(

        documents: List[str],

        queries: List[str],

) -> List[str]:
    """Finds the closest document for each query."""

    N = len(documents)
    # 1. First step is to preprocess the documents
    processed_documents = []
    for i in range(N):
        text = documents[i]
        processed_documents.append(word_tokenize(str(preprocess(text))))

    # 2. Calculating document frequency for all words
    doc_freq = {}
    for i in range(N):
        tokens = processed_documents[i]
        for w in tokens:
            if w in doc_freq:
                doc_freq[w] = doc_freq[w] + 1
            else:
                doc_freq[w] = 1
    # 3. Saving the words vocabulary in our documents for further use
    vocabulary = [x for x in doc_freq]

    # 4. Generating tf-idf matrix
    tf_idf = {}
    for i in range(N):
        token_dict = Counter(processed_documents[i])
        words_count = len(processed_documents[i])
        for token in np.unique(processed_documents[i]):
            tf = token_dict[token] / words_count
            df = get_document_freq(doc_freq, token)
            idf = np.log((N + 1) / (df + 1))
            tf_idf[i, token] = tf * idf

    # 5. Generating tf-idf array
    tf_idf_mat = np.zeros((N, len(vocabulary)))
    for i in tf_idf:
        ind = vocabulary.index(i[1])
        tf_idf_mat[i[0]][ind] = tf_idf[i]

    # Generate list of similar documents using cosine similarity

    similar_doc = []
    for i in queries:
        preprocessed_query = preprocess(i)
        doc_cosines = []
        query_vector = vector_encoder(preprocessed_query, vocabulary, N)
        for doc in tf_idf_mat:
            doc_cosines.append(cosine_sim(query_vector, doc))
        doc_id = np.array(doc_cosines).argsort()[::-1][0]
        similar_doc.append(documents[doc_id])

    return similar_doc


def vector_encoder(data: str, total_vocab: List[str], N: int, doc_freq: Dict[str, int]) -> np.ndarray:
    """Gives the corresponding vector representation of a query"""
    tokens = word_tokenize(str(data))
    Q_vec = np.zeros((len(total_vocab)))
    counter = Counter(tokens)
    words_count = len(tokens)
    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = get_document_freq(doc_freq, token)
        idf = np.log((N + 1) / (df + 1))
        try:
            ind = total_vocab.index(token)
            Q_vec[ind] = tf * idf
        except:
            pass
    return Q_vec


def cosine_sim(v1: np.ndarray, v2: np.ndarray):
    """Returns cosine similarity"""
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim


def convert_lower_case(data: str) -> str:
    """Convert all letters in a string to lowercase."""
    return np.char.lower(data)


def remove_punctuation(data: str) -> str:
    """Removes all the punctuations from a string."""
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    data = np.char.replace(data, "'", "")
    data = np.char.replace(data, ".", "")
    return data


def remove_stop_words(data: str) -> str:
    """Removes stop words from a string."""
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def lemma_string(data: str) -> str:
    """Lemmatize words in a string."""
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemma.lemmatize(w)
    return new_text


def preprocess(text: str) -> str:
    """Preprocess a string using a set of defined rules."""
    text = convert_lower_case(text)
    text = remove_punctuation(text)
    text = remove_stop_words(text)
    text = lemma_string(text)
    return text


def get_document_freq(doc_freq, word:str):
    f = 0
    try:
        f = doc_freq[word]
    except:
        pass
    return f
