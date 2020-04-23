import re
import joblib
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
stop = stopwords.words('english')

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']


# Allows input from multiple sources
def input_stream(doc):
    pass


# Document cleaning
def cleanup(document):
    """Cleans document from unneccessary characters"""
    document = re.sub('[^A-Za-z0-9 .-?!#",]+', ' ', document)
    document = merge_acronyms(document)
    clean_document = ' '.join(document.split())
    return clean_document


def merge_acronyms(s):
    """Merges all acronyms in a given sentence. For example G.T. -> Sucks """
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.', ''))
    return s


def rank_sentences(document, dense_matrix, features_matrix, top_n=3):
    """Ranks the most-relevant sentences and returns the top n-number of sentences in the order it appears in the article"""
    sentence_list = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentence_list]
    clean_sentences = [[word for word in sent if nltk.pos_tag([word])[0][1] in NOUNS]
                       for sent in sentences]
    tfidf_sentences = [
        [dense_matrix[features_matrix.index(word.lower())] for word in sent if word.lower() in features_matrix]
        for sent in clean_sentences]

    doc_value = sum(dense_matrix)
    sentences_value = [sum(sent) / doc_value for sent in tfidf_sentences]
    sentences_value = np.array(sentences_value)
    # TODO: Continue/Improve Sentence value scoring algorithm

    # weighted_sent = [sent*(1 - i/(2*len(sentences_value))) for i, sent in enumerate(sentences_value)]
    ranked_sent = [pair for pair in zip(range(len(sentences_value)), sentences_value)]
    sorted_sent = sorted(ranked_sent, key=lambda x: x[1] * -1)  # Sentence 30, 8, 1
    return sorted_sent[:top_n]


def doc_tfidf_dense(document, pipeline):
    doc_freq_term = pipeline['count'].transform([document])
    doc_tfidf_matrix = pipeline['tfid'].transform(doc_freq_term)
    dense_matrix = doc_tfidf_matrix.toarray().tolist()[0]
    return doc_tfidf_matrix, dense_matrix


def summary(top_sentences, document):
    index_sentences = sorted([top_sentences[i][0] for i in range(len(top_sentences))])
    sentence_list = nltk.sent_tokenize(document)
    summary_list = [sentence_list[i] for i in index_sentences]
    return ' '.join(summary_list)


# TODO - Preprocessing & Optimization for larger datasets
if __name__ == "__main__":
    df = pd.read_csv('articles1.csv')
    data = df.drop(columns=['Unnamed: 0', 'id', 'publication', 'author', 'date', 'year', 'month', 'url'])
    # document = cleanup(data.values[10002, 1]) # Make sure to make this input unique & robust
    document = data.values[100, 1]
    corpus = data.values[:10005, 1]  # Pre-process and optimize

    pipe = Pipeline([('count', CountVectorizer()),
                     ('tfid', TfidfTransformer())]).fit(corpus)

    joblib.dump(pipe['count'], 'model/count_vectorizer_model.pkl', compress=True)
    joblib.dump(pipe['tfid'], 'model/tfid_transformer_model.pkl', compress=True)

    features_matrix = pipe['count'].get_feature_names()
    doc_tfidf, dense_matrix = doc_tfidf_dense(document, pipe)

    top_sentences = rank_sentences(document, dense_matrix, features_matrix, top_n=4)
    summary_text = summary(top_sentences, document)
    with open("summarized.txt", 'w') as summ, open("original.txt", 'w') as doc:
        summ.write(summary_text)
        doc.write(document)
