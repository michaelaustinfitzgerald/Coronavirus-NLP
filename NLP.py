import codecs
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import re
import pandas as pd
import plotly.graph_objects as go
from imageio import imread
from wordcloud import STOPWORDS, WordCloud
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv(r'Corona_NLP_train.csv')
test = pd.read_csv(r'Corona_NLP_test.csv')

# WordCloud code
df_wc = pd.concat([train['OriginalTweet'], test['OriginalTweet']]).values
f = open(r'C:\Users\micha\Desktop\NLP\covid.txt', 'r')  # opening a binary file
covid = f.read()
covid_64 = str.encode(covid)
f1 = open("pfizer.png", "wb")
f1.write(codecs.decode(covid_64, 'base64'))
f1.close()
img = imread("pfizer.png")
hccovid = img
stopwords = set(STOPWORDS)
stopwords = stopwords.union({'HTTPS', 'T'})
plt.figure(figsize=(16, 13))
wc = WordCloud(background_color="black",
               max_words=10000,
               mask=hccovid,
               stopwords=stopwords,
               max_font_size=40)
wc.generate(" ".join(df_wc))
plt.title("Coronavirus Tweets WordCloud", fontsize=20)
plt.imshow(wc.recolor(colormap='Pastel2', random_state=37))
plt.axis('off')
# plt.show()

# Sentiment EDA
sns.set_theme()
sns.countplot(train['Sentiment'])
# plt.show()

# Remapping sentiment for Extremely Positive/Extremely Negative
train = train[['OriginalTweet', 'Sentiment']]
test = test[['OriginalTweet', 'Sentiment']]


def simple_sentiment(sent):
    if sent == 'Extremely Positive':
        return 'Positive'
    elif sent == 'Extremely Negative':
        return 'Negative'
    else:
        return sent


train['Sentiment'] = train['Sentiment'].apply(lambda x: simple_sentiment(x))
test['Sentiment'] = test['Sentiment'].apply(lambda x: simple_sentiment(x))

train = train[train.Sentiment != "Neutral"]

sns.set_theme()
sns.countplot(train['Sentiment'])

# plt.show()


#Clean data to remove stop words, mentions, urls and html
def clean_data(text):
    # removes any words attached to @ symbol
    text = re.sub(r'@\S+', ' ', text)
    # removes everything after http up until a space
    text = re.sub(r'http\S+', ' ', text)
    # removes html tags
    text = re.sub(r'<.*?>', ' ', text)
    # removes hashtags
    text = re.sub(r'#\S+', ' ', text)
    text = text.lower()
    # removing stop words
    text = text.split()
    text = " ".join([
        word for word in text
        if not word in nltk.corpus.stopwords.words('english')
    ])
    return text


train['OriginalTweet'] = train['OriginalTweet'].apply(lambda x: clean_data(x))
test['OriginalTweet'] = test['OriginalTweet'].apply(lambda x: clean_data(x))
all_words_train = train['OriginalTweet'].str.split(
    expand=True).unstack().value_counts()
all_words_test = test['OriginalTweet'].str.split(
    expand=True).unstack().value_counts()
all_words = pd.concat([all_words_train, all_words_test])
data = [
    go.Bar(x=all_words.index.values[2:50],
           y=all_words.values[2:50],
           marker=dict(colorscale='Jet', color=all_words.values[2:100]),
           text='Word Counts')
]
layout = go.Layout(title='Top 50 (uncleaned) words in the training data')
fig = go.Figure(data=data, layout=layout)
fig.show()

# Encoding
label_encoder = LabelEncoder()
train['Sentiment'] = label_encoder.fit_transform(train['Sentiment'])
test['Sentiment'] = label_encoder.fit_transform(test['Sentiment'])

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


train['OriginalTweet'] = train['OriginalTweet'].apply(
    lambda x: lemmatize_text(x))


# https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
# Dummy function to deal with already tokenized data
def dummy(doc):
    return doc


tfidf = CountVectorizer(
    tokenizer=dummy,
    preprocessor=dummy,
)
text_counts = tfidf.fit_transform(train['OriginalTweet'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_counts,
                                                    train['Sentiment'],
                                                    test_size=0.3,
                                                    random_state=1)

nb = MultinomialNB()
grid_params = {
    'alpha': np.linspace(0.5, 1.5, 11),
    'fit_prior': [True, False],
}

clf = GridSearchCV(nb, grid_params)
clf.fit(X_train, y_train)
print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)

nb = MultinomialNB(alpha=1.5, fit_prior=False).fit(X_train, y_train)
predicted = nb.predict(X_test)
print("MultinomialNB accuracy", metrics.accuracy_score(y_test, predicted))
cm = metrics.confusion_matrix(y_test, predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test,
                                         nb.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
# plt.show()
