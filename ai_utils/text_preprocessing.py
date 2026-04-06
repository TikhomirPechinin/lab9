import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Входное значение должно быть строкой")
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
    return text.lower().strip()


def remove_stopwords(text: str, language: str = 'english') -> str:
    if not isinstance(text, str):
        raise TypeError("Входное значение должно быть строкой")
    try:
        stop_words = set(stopwords.words(language))
    except:
        stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


def stem_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Входное значение должно быть строкой")
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def tokenize_text(text: str) -> list:
    if not isinstance(text, str):
        raise TypeError("Входное значение должно быть строкой")
    from nltk.tokenize import word_tokenize
    return word_tokenize(text.lower())