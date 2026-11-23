# NLP Disaster Tweets Classification

Бинарная классификация твитов на тему стихийных бедствий  

## данные

- **train.csv / test.csv**: https://drive.google.com/drive/folders/1RH2FBAtNoXrCT8Mm_dNszqTtyef_ubwP?usp=sharing
- нужно положить в корневую папку проекта

## обученные модели
- обученные модели по всем экспериментам: https://drive.google.com/drive/folders/1P478p281Goo3ntVvFSQRcmXiVer_cvix?usp=sharing
- загружать в папку nlp_disaster_tweets/models 

## структура

```
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                  # yaml-конфиги экспериментов
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── classic_ml.yaml
│   ├── embeddings.yaml
│   └── finetune.yaml
├── notebooks/                # чуть-чуть eda
│   └── 01_eda.ipynb
├── scripts/
│   ├── train_classic.py      # logreg, catboost с разным препроцессингом и векторизаторами
│   ├── train_embeddings.py   # logreg на эмбеддингах, полученных от sentence-transformers/all-MiniLM-L6-v2
│   ├── train_finetune.py     # fine_tunning distilbert
│   └── evaluate.py           # оценка качества всех моделей из папки models
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── clean.py          # препроцессинг
│   ├── features/
│   │   ├── __init__.py
│   │   └── embeddings.py     # получением эмбеддингов sentence-transformers/all-MiniLM-L6-v2
│   ├── models/
│   └── utils/
│       ├── __init__.py
│       └── seed.py           # для воспроизводимости
└── models/
```
## установка

```bash
pip install -r requirements.txt
```

## РЕЗУЛЬТАТЫ

## tf-idf + logreg/catboost + preprocessing(v1/v2/v3)

| Подход | Модель | Препроцессинг | Val F1 | Val Accuracy |


| tf-idf | logreg |      v1       | 0.7624 |    0.7958    |

| tf-idf | logreg |      v2       | 0.7677 |    0.8017    |

| tf-idf | logreg |      v3       | 0.7630 |    0.8017    |


| tf-idf | catboost |    v1       | 0.7494 |    0.7971    |

| tf-idf | catboost |    v2       | 0.7496 |    0.7991    |

| tf-idf | catboost |    v3       | 0.7368 |    0.7932    |

- с увеличением препроцессинга качество улучшается
- логистическая регрессия по всем экспериментам лучше результат показала
- после tf-idf вектора разреженные, из за этого наверное catboost хуже

## word2vec + logreg/catboost + preprocessing(v1/v2/v3)

| Подход | Модель | Препроцессинг | Val F1 | Val Accuracy |


|   w2v  | logreg |      v1       | 0.6975 |    0.7249    |

|   w2v  | logreg |      v2       | 0.6917 |    0.7196    |

|   w2v  | logreg |      v3       | 0.7018 |    0.7321    |


|   w2v  | catboost |    v1       | 0.6499 |    0.7170    |

|   w2v  | catboost |    v2       | 0.6023 |    0.6756    |

|   w2v  | catboost |    v3       | 0.6313 |    0.6986    |

w2v брала из gensim и обучала на train части датасета(предварительно отделив val часть)
и данных для w2v, видимо, маловато оказалось. по метрикам уступает tf-idf 

с v2(базовая очистка + удаление стоп-слов) результат хуже всех и у logreg, и у catboost.
возможно, так произошло, из за того, что твиты сами по себе короткие, после удаления стоп-слов это усугубляется? а эмбеддинги из word2vec я усредняла, чтобы получить эмбеддинги для твитов. 
скорее всего, они слишком шумные получились 

а добавление лемматизации в v3 исправило ситуацию и качество улучшилось на обеиз моделях

## miniLM + logreg + v1/без препроцессинга

| Подход | Модель | Препроцессинг | Val F1 | Val Accuracy |


| miniLM | logreg |       -       | 0.7964 |    0.8221    |

| miniLM | logreg |       v1      | 0.7812 |    0.8043    |

[huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) модельку отсюда брала. при выборе смотрела на колво параметров и качество с лидерборда.
и + у нее в тренировочных данных самая большая доля - комментарии с reddit, то есть тоже короткие тексты в разговорном стиле как у меня в задаче

и хотела посмотреть как базовая очистка повлияет. чат gpt не соврал, правда лучше без нее

## distilbert 

|   Модель    | Препроцессинг | Val F1 | Val Accuracy |


| DistilBERT  |        -      | 0.8203 |    0.8444    |

как и ожидалось, тут метрики выше всего
эту модельку тоже с hugging face брала. ну и в целом тоже смотрела на количество параметров и качество.
+ изначально bert-подобную модель искала, чтобы по cls представлению классификацию делать 


## запуск

```bash
# classic ml
python scripts/train_classic.py --mode tfidf --preprocess v1 --model logreg
python scripts/train_classic.py --mode word2vec --preprocess v1 --model catboost

# miniLM + logreg
python scripts/train_embeddings.py
python scripts/train_embeddings.py --use-preprocessed

# fine-tunning distilbert
python scripts/train_finetune.py

# оценка
python scripts/evaluate.py
```

## что можно улучшить

- можно попробовать более крупные модели
- добавить аугментацию текста
- еще ансамбли из лучших моделек для kaggle можно попробовать сделать 
