import re
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords; stop = stopwords.words('english')



NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

# TODO - Preprocessing & Optimization for larger datasets
df = pd.read_csv('articles1.csv')
data = df.drop(columns=['Unnamed: 0', 'id', 'publication', 'author', 'date', 'year', 'month', 'url'])
corpus = data.values[:1000, 1]
document = data.values[10, 1]
vectorizer =  CountVectorizer()
freq_term_matrix = vectorizer.fit_transform(corpus)
features_matrix = vectorizer.get_feature_names()

tfidf = TfidfTransformer(norm='l2')
tfidf.fit(freq_term_matrix)

doc_freq_term = vectorizer.transform([document])
doc_tfidf_matrix = tfidf.transform(doc_freq_term)
dense_matrix = doc_tfidf_matrix.toarray().tolist()[0]

sentence_list = nltk.sent_tokenize(document)
sentences = [nltk.word_tokenize(sent) for sent in sentence_list]
clean_sentences = [[word for word in sent if nltk.pos_tag([word])[0][1] in NOUNS]
                  for sent in sentences]

tfidf_sentences = [[dense_matrix[features_matrix.index(word.lower())] for word in sent if word.lower() in features_matrix]
                    for sent in clean_sentences]

doc_value = sum(dense_matrix)
sentences_value = [sum(sent) / doc_value for sent in tfidf_sentences]
sentences_value = np.array(sentences_value)
# TODO: Continue/Improve Sentence value scoring algorithm

weighted_sent = [sent*(1 - i/(2*len(sentences_value))) for i, sent in enumerate(sentences_value)]
ranked_sent = [pair for pair in zip(range(len(weighted_sent)), weighted_sent)]
sorted_sent = sorted(ranked_sent, key = lambda x: x[1] * -1) # Sentence 30, 8, 1
top_sentences = sorted_sent[:4]

summarizer = sorted(top_sentences, key = lambda x: x[0])

sentenceSummary = [sentence_list[i] for i, value in summarizer]
