import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# v1 - базовая очистка
def clean_v1(text):
    return basic_clean(text)

# v2 - v1 + удаление стоп-слов
def clean_v2(text):
    text = basic_clean(text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# v3 - v2 + лемматизация
def clean_v3(text):
    text = basic_clean(text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def apply_preprocessing(texts, version: str = "v1"):
    clean_fn = {"v1": clean_v1, "v2": clean_v2, "v3": clean_v3}[version]
    return [clean_fn(t) for t in texts]