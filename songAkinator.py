import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier


def akinate(noise, lemmatizer, vectorizer, clf, song):
    data = pd.DataFrame([song], columns=['lyrics'])
    lowered = data['lyrics'].str.lower()
    data['lowered'] = lowered
    tokened = data.apply(lambda row: nltk.word_tokenize(row['lowered']), axis=1)
    data['tokened'] = tokened
    withoutstop = data['tokened'].apply(lambda x: [item for item in x if item not in noise])
    without_stop = []
    for a in withoutstop:
        without_stop.append(", ".join(a))
    data['without_stop'] = without_stop
    lemmatized = data['without_stop'].apply(lambda x: [lemmatizer.lemmatize(x)])
    lemma = []
    for a in lemmatized:
        lemma.append(", ".join(a))
    data['lemmatized'] = lemma
    x_test = data.lemmatized
    vectorized_x_test = vectorizer.transform(x_test)
    pred = clf.predict(vectorized_x_test)
    return pred[0]


def learn():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    data = pd.read_csv('tcc_ceds_music.csv')
    columns = data[['genre', 'lyrics']]
    lowered = columns['lyrics'].str.lower()
    columns['lowered'] = lowered

    tokened = columns.apply(lambda row: nltk.word_tokenize(row['lowered']), axis=1)
    columns['tokened'] = tokened

    noise = stopwords.words('english')
    withoutstop = columns['tokened'].apply(lambda x: [item for item in x if item not in noise])
    without_stop = []
    for a in withoutstop:
        without_stop.append(", ".join(a))
    columns['without_stop'] = without_stop

    lemmatizer = WordNetLemmatizer()
    lemmatized = columns['without_stop'].apply(lambda x: [lemmatizer.lemmatize(x)])
    lemma = []
    for a in lemmatized:
        lemma.append(", ".join(a))
    columns['lemmatized'] = lemma

    x_train, x_test, y_train, y_test = train_test_split(columns.lemmatized, columns.genre, train_size=0.7)
    columns.genre.value_counts()

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    vectorized_x_train = vectorizer.fit_transform(x_train)

    clf = RidgeClassifier(alpha=30)
    clf.fit(vectorized_x_train, y_train)
    vectorized_x_test = vectorizer.transform(x_test)
    pred = clf.predict(vectorized_x_test)
    print(classification_report(y_test, pred))
    return noise, lemmatizer, vectorizer, clf


if __name__ == '__main__':
    learn()
    pass
