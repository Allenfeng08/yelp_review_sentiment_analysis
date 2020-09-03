# Yelp dataset sentiment analysis
import string
import re
import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as sw
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy.sparse import vstack
from nltk.tag import pos_tag
from langdetect import detect
from textblob import TextBlob
import multiprocessing
import matplotlib.pyplot as plt
import swifter
import pickle
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
from tqdm import tqdm
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# read the data in csv format
df_review = pd.read_csv("/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/yelp-dataset_csv/yelp_academic_dataset_review.csv")
print(df_review.iloc[0])
print(df_review.columns)

# convert the datatype of stars to category
df_review['stars'].astype('category')

# Select the 'text' and 'stars' columns from the review dataset
df_review_1 = df_review[['text','stars']]
# Drop the rows with nan
df_review_2 = df_review_1.dropna()
# The shape of df_review_2: [8021120 rows x 2 columns]

# Save df_review_1
df_review_2.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_2.pkl')

# load df_review_1
df_review_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_2.pkl')

# Split the df_review_loaded into three parts for preprocessing
df_review_loaded_1 = df_review_loaded.iloc[:3000000]
df_review_loaded_1.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_1.pkl')
df_review_loaded_1_new = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_1.pkl')

df_review_loaded_2 = df_review_loaded.iloc[3000000:6000000]
df_review_loaded_2.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_2.pkl')
df_review_loaded_2_new = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_2.pkl')

df_review_loaded_3 = df_review_loaded.iloc[6000000:]
df_review_loaded_3.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_3.pkl')
df_review_loaded_3_new = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_3.pkl')

# Preprocessing of the reviews
# 1. Convert to lower case
# 2. Tokenization
# 3. Remove punctuations and specific symbols
# 4. Filter with stop words
# 5. Lemmatization

# Display all the possible tags
nltk.help.upenn_tagset()

def preprocessing(review):
    # Convert to lower case
    df_review_1_lower = review.lower()
    # Tokenization
    tokenized_text = word_tokenize(df_review_1_lower)
    # Remove punctuations and specific symbols
    tokenized_text_removed = []
    for word in tokenized_text:
        if word not in string.punctuation and word.isalpha():
            tokenized_text_removed.append(word)
    # Filter with stop words
    stopwords = sw.words('english')
    tokenized_text_filtered = []
    for word in tokenized_text_removed:
        if word not in stopwords:
            tokenized_text_filtered.append(word)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    token = []
    for word, tag in pos_tag(tokenized_text_filtered):
        if tag.startswith("NN"):
            word_lemmatized = lemmatizer.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            word_lemmatized = lemmatizer.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            word_lemmatized = lemmatizer.lemmatize(word, pos='a')
        else:
            word_lemmatized = word
        token.append(word_lemmatized)
    text = ' '.join(token)

    if len(text.strip()) != 0 and detect(text) == 'en':
        text_1 = text
    else:
        text_1 = np.NaN
    return [text_1], token

# Preprocessing the entire dataset
# set_npartitions(n): select number of cores

# Preprocessing of part 1
start = time.perf_counter()
df_review_loaded_1_new['text_processed'] = df_review_loaded_1_new['text'].swifter.set_npartitions(8).allow_dask_on_strings(enable=True).apply(preprocessing)
finish = time.perf_counter()
df_review_loaded_1_new.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_1_new.pkl')
df_review_part_1 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_1_new.pkl')
df_review_part_1 = df_review_part_1.drop('text', axis=1)
df_review_part_1.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_1.pkl')
df_review_part_1_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_1.pkl')
df_review_part_1_loaded[['text', 'token']] = pd.DataFrame(df_review_part_1_loaded['text_processed'].tolist(), index=df_review_part_1_loaded.index)
df_review_part_1_loaded = df_review_part_1_loaded.drop('text_processed', axis=1)
df_review_part_1_loaded['stars'] = df_review_part_1_loaded['stars'].astype('category')
df_review_part_1_loaded.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_1_final.pkl')
df_review_seg_1 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_1_final.pkl')
df_review_seg_1['text_new'] = pd.DataFrame(df_review_seg_1['text'].tolist(), index=df_review_seg_1.index)
df_review_seg_1 = df_review_seg_1.dropna()
df_review_seg_1.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_seg_1.pkl')
df_review_data_1 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_seg_1.pkl')
df_review_data_1 = df_review_data_1.drop(columns=['text_new'])
df_review_data_1.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_data_1.pkl')
print(f'Finished in {round(finish-start,2)} second(s)')

# Preprocessing of part 2
df_review_loaded_2_new['text_processed'] = df_review_loaded_2_new['text'].swifter.set_npartitions(8).allow_dask_on_strings(enable=True).apply(preprocessing)
df_review_loaded_2_new.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_2_new.pkl')
df_review_part_2 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_2_new.pkl')
df_review_part_2 = df_review_part_2.drop('text', axis=1)
df_review_part_2.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_2.pkl')
df_review_part_2_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_2.pkl')
df_review_part_2_loaded[['text', 'token']] = pd.DataFrame(df_review_part_2_loaded['text_processed'].tolist(), index=df_review_part_2_loaded.index)
df_review_part_2_loaded = df_review_part_2_loaded.drop('text_processed', axis=1)
df_review_part_2_loaded['stars'] = df_review_part_2_loaded['stars'].astype('category')
df_review_part_2_loaded.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_2_final.pkl')
df_review_seg_2 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_2_final.pkl')
df_review_seg_2['text_new'] = pd.DataFrame(df_review_seg_2['text'].tolist(), index=df_review_seg_2.index)
df_review_seg_2 = df_review_seg_2.dropna()
df_review_seg_2.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_seg_2.pkl')
df_review_data_2 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_seg_2.pkl')
df_review_data_2 = df_review_data_2.drop(columns=['text_new'])
df_review_data_2.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_data_2.pkl')

# Preprocessing of part 3
df_review_loaded_3_new['text_processed'] = df_review_loaded_3_new['text'].swifter.set_npartitions(8).allow_dask_on_strings(enable=True).apply(preprocessing)
df_review_loaded_3_new.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_3_new.pkl')
df_review_part_3 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_loaded_3_new.pkl')
df_review_part_3 = df_review_part_3.drop('text', axis=1)
df_review_part_3.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_3.pkl')
df_review_part_3_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_3.pkl')
df_review_part_3_loaded[['text', 'token']] = pd.DataFrame(df_review_part_3_loaded['text_processed'].tolist(), index=df_review_part_3_loaded.index)
df_review_part_3_loaded = df_review_part_3_loaded.drop('text_processed', axis=1)
df_review_part_3_loaded['stars'] = df_review_part_3_loaded['stars'].astype('category')
df_review_part_3_loaded.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_3_final.pkl')
df_review_seg_3 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_part_3_final.pkl')
df_review_seg_3['text_new'] = pd.DataFrame(df_review_seg_3['text'].tolist(), index=df_review_seg_3.index)
df_review_seg_3 = df_review_seg_3.dropna()
df_review_seg_3.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_seg_3.pkl')
df_review_data_3 = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_seg_3.pkl')
df_review_data_3 = df_review_data_3.drop(columns=['text_new'])
df_review_data_3.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_data_3.pkl')


##################### Analysis starting from here
# Analyze the first 1 million preprocessed reviews
df_review_preprocessed = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_one_million.pkl')

# Split the dataset into train and test
df_review_train = df_review_preprocessed.iloc[0:int(0.8*len(df_review_preprocessed))]
df_review_test = df_review_preprocessed.iloc[int(0.8*len(df_review_preprocessed)):]

df_review_train.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_train.pkl')
df_review_test.to_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_test.pkl')

df_review_train_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_train.pkl')
df_review_test_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_test.pkl')

# Combine all the tokens of the 1 million reviews
all_token = []
for token in df_review_train_loaded['token']:
    all_token += token

# Choose the top 5000 topwords
features = FreqDist(all_token)
topwords = [word[0] for word in list(features.most_common(5000))]
print(topwords[-1])
# The 5000th topword is 'expected'
print(features[topwords[-1]])
# 'expected' occurs 544 times in all_token

sum_count = 0

for word in topwords:
    sum_count += features[word]

# The length of all_token is 42911317
# The totol word count for the top 5000 topwords is 40205576
# Which represent more than 90% of the all tokens

# Save the feature and
with open("/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/features.txt", "wb") as fp:
    pickle.dump(features, fp)
# loading
with open("/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/features.txt", "rb") as fp:
    features_loaded = pickle.load(fp)

# Save the topwords
with open("/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/topwords.txt", "wb") as fa:
    pickle.dump(topwords, fa)
# loading
with open("/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/topwords.txt", "rb") as fa:
    topwords_loaded = pickle.load(fa)

# Create the corpus
corpus = []
for review in df_review_train_loaded['text']:
    corpus += review

# Create y_train
label = []
for star in df_review_train_loaded['stars']:
    label += [star]

## Navie Bayes
# Create X_train using count vectorizer
count_vector = CountVectorizer()
count_vector.fit(topwords_loaded)
train_X_count = count_vector.transform(corpus)
train_X_count = train_X_count.toarray()

# Create X_train using tfidf vectorizer
tfidf_vector = TfidfVectorizer()
tfidf_vector.fit(topwords_loaded)
train_X_tfidf = tfidf_vector.transform(corpus)
train_X_tfidf = train_X_tfidf.toarray()

# Fit model using train_X_count
# MultinomialNB
model_MN_count = MultinomialNB()
model_MN_count.fit(train_X_count, label)

# Save model
model_MN_count_filename = "/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/model_MN_count.pkl"
with open(model_MN_count_filename, 'wb') as file:
    pickle.dump(model_MN_count, file)
# Load model
with open(model_MN_count_filename, 'rb') as file:
    model_MN_count_load = pickle.load(file)

# GaussianNB
model_Gau_count = GaussianNB()
model_Gau_count.fit(train_X_count, label)

# Save model
model_Gau_count_filename = "/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/model_Gau_count.pkl"
with open(model_Gau_count_filename, 'wb') as file:
    pickle.dump(model_Gau_count, file)
# Load model
with open(model_Gau_count_filename, 'rb') as file:
    model_Gau_count_load = pickle.load(file)


# Fit model using train_X_tfidf
# MultinomialNB
model_MN_tfidf = MultinomialNB()
model_MN_tfidf.fit(train_X_tfidf, label)

model_MN_tfidf_filename = "/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/model_MN_tfidf.pkl"
with open(model_MN_tfidf_filename, 'wb') as file:
    pickle.dump(model_MN_tfidf, file)

with open(model_MN_tfidf_filename, 'rb') as file:
    model_MN_tfidf_load = pickle.load(file)

# GaussianNB
model_Gau_tfidf = GaussianNB()
model_Gau_tfidf.fit(train_X_tfidf, label)

model_Gau_tfidf_filename = "/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/model_Gau_tfidf.pkl"
with open(model_Gau_tfidf_filename, 'wb') as file:
    pickle.dump(model_Gau_tfidf, file)

with open(model_Gau_tfidf_filename, 'rb') as file:
    model_Gau_tfidf_load = pickle.load(file)

# Testing strats here
corpus_test = []
for review in df_review_test_loaded['text']:
    corpus_test += review

label_test = []
for star in df_review_test_loaded['stars']:
    label_test += [star]

count_vector_1 = CountVectorizer()
count_vector_1.fit(topwords_loaded)
test_X_count = count_vector_1.transform(corpus_test)
test_X_count = test_X_count.toarray()

tfidf_vector_1 = TfidfVectorizer()
tfidf_vector_1.fit(topwords_loaded)
test_X_tfidf = tfidf_vector_1.transform(corpus_test)
test_X_tfidf = test_X_tfidf.toarray()

prediction_MN_count = model_MN_count_load.predict(test_X_count)
prediction_MN_tfidf = model_MN_tfidf_load.predict(test_X_tfidf)
prediction_Gau_count = model_Gau_count_load.predict(test_X_count)
prediction_Gau_tfidf = model_Gau_tfidf_load.predict(test_X_tfidf)

np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/prediction_MN_count.npy', prediction_MN_count)
np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/prediction_MN_tfidf.npy', prediction_MN_tfidf)
np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/prediction_Gau_count.npy', prediction_Gau_count)
np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/prediction_Gau_tfidf.npy', prediction_Gau_tfidf)

print(accuracy_score(prediction_MN_count, label_test))
print(accuracy_score(prediction_MN_tfidf, label_test))
print(accuracy_score(prediction_Gau_count, label_test))
print(accuracy_score(prediction_Gau_tfidf, label_test))

print(classification_report(prediction_MN_count, label_test))
print(classification_report(prediction_MN_tfidf, label_test))
print(classification_report(prediction_Gau_count, label_test))
print(classification_report(prediction_Gau_tfidf, label_test))
########################### End of Navie Bayes



########################### Start of Neural Nets
df_review_preprocessed = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_one_million.pkl')
df_review_train_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_train.pkl')
df_review_test_loaded = pd.read_pickle('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/df_review_test.pkl')

label_test = []
for star in df_review_test_loaded['stars']:
    label_test += [star]


# label for training
y_train = df_review_train_loaded['stars'][0:600000]
y_train_final = []
for rating in y_train:
    y_train_final.append(rating)
y_train_final = pd.DataFrame(y_train_final)

# label for validation
y_validation = df_review_train_loaded['stars'][600000:800000]
y_validation_final = []
for rating in y_validation:
    y_validation_final.append(rating)
y_validation_final = pd.DataFrame(y_validation_final)


###########################
# Neural Networks - Doc2Vec/DBOW (Distributed Bag Of Words)
def label_review(review, label):
    result = []
    prefix = label
    for i, t in zip(review.index, review):
        result.append(TaggedDocument(t, [prefix + '_%s' % i]))
    return result

test_1 = label_review(df_review_preprocessed['token'], 'test')

cores = multiprocessing.cpu_count()
gensim_model_d2v_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
gensim_model_d2v_dbow.build_vocab([x for x in tqdm(test_1)])

for epoch in range(20):
    gensim_model_d2v_dbow.train(utils.shuffle([x for x in tqdm(test_1)]), total_examples=len(test_1), epochs=1)
    gensim_model_d2v_dbow.alpha -= 0.002
    gensim_model_d2v_dbow.min_alpha = gensim_model_d2v_dbow.alpha

gensim_model_d2v_dbow_filename = "/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/gensim_model_d2v_dbow.pkl"
'''
# Saving
with open(gensim_model_d2v_dbow_filename, 'wb') as file:
    pickle.dump(gensim_model_d2v_dbow, file)
'''
# Loading
with open(gensim_model_d2v_dbow_filename, 'rb') as file:
    gensim_model_d2v_dbow_load = pickle.load(file)


def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'test_' + str(i)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs

train_vecs_dbow = get_vectors(gensim_model_d2v_dbow_load, df_review_train_loaded['text'][0:600000], 100)
validation_vecs_dbow = get_vectors(gensim_model_d2v_dbow_load, df_review_train_loaded['text'][600000:800000], 100)
test_vecs_dbow = get_vectors(gensim_model_d2v_dbow_load, df_review_test_loaded['text'], 100)

# Modeling of the neural nets
seed = 100
np.random.seed(seed)
model_dbow = Sequential()
model_dbow.add(Dense(50, activation='relu', input_dim=100))
model_dbow.add(Dense(6, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.0002)
model_dbow.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_dbow.fit(train_vecs_dbow,
               y_train_final,
               validation_data=(validation_vecs_dbow, y_validation_final),
               epochs=5,
               batch_size=10,
               verbose=2)

model_dbow.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/model_dbow')

prediction_model_dbow = np.argmax(model_dbow.predict(test_vecs_dbow), axis=-1)
np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/prediction_model_dbow.npy', prediction_model_dbow)

print(accuracy_score(prediction_model_dbow, label_test))
print(classification_report(prediction_model_dbow, label_test))


##########################################
# Neural Network - Doc2Vec/DM (Distributed Memory)
cores = multiprocessing.cpu_count()
gensim_model_d2v_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=5, min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
gensim_model_d2v_dmm.build_vocab([x for x in tqdm(test_1)])

for epoch in range(20):
    gensim_model_d2v_dmm.train(utils.shuffle([x for x in tqdm(test_1)]), total_examples=len(test_1), epochs=1)
    gensim_model_d2v_dmm.alpha -= 0.002
    gensim_model_d2v_dmm.gensim_model_d2v_dmm = model_dmm.alpha


gensim_model_d2v_dmm_filename = "/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/gensim_model_d2v_dmm.pkl"

with open(gensim_model_d2v_dmm_filename, 'rb') as file:
    gensim_model_d2v_dmm_load = pickle.load(file)


train_vecs_dmm = get_vectors(gensim_model_d2v_dmm_load, df_review_train_loaded['text'][0:600000], 100)
validation_vecs_dmm = get_vectors(gensim_model_d2v_dmm_load, df_review_train_loaded['text'][600000:800000], 100)
test_vecs_dmm = get_vectors(gensim_model_d2v_dmm_load, df_review_test_loaded['text'], 100)

# Modeling of the neural nets
seed = 100
np.random.seed(seed)
model_dmm = Sequential()
model_dmm.add(Dense(50, activation='relu', input_dim=100))
model_dmm.add(Dense(6, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.0002)
model_dmm.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_dmm.fit(train_vecs_dmm,
              y_train_final,
              validation_data=(validation_vecs_dmm, y_validation_final),
              epochs=5,
              batch_size=10,
              verbose=2)

model_dmm.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/model_dmm_01')

prediction_model_dmm = np.argmax(model_dmm.predict(test_vecs_dmm), axis=-1)
np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/prediction_model_dmm.npy', prediction_model_dmm)

print(accuracy_score(prediction_model_dmm, label_test))
print(classification_report(prediction_model_dmm, label_test))


##########################################
# Concatenate the above two models
model_merge = ConcatenatedDoc2Vec([gensim_model_d2v_dbow_load, gensim_model_d2v_dmm_load])
train_vecs_merge = get_vectors(model_merge, df_review_train_loaded['text'][0:600000], 200)
validation_vecs_merge = get_vectors(model_merge, df_review_train_loaded['text'][600000:800000], 200)
test_vecs_merge = get_vectors(model_merge, df_review_test_loaded['text'], 200)

# Modeling of the neural nets
seed = 100
np.random.seed(seed)
model_merge = Sequential()
model_merge.add(Dense(50, activation='relu', input_dim=200))
model_merge.add(Dense(6, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.0002)
model_merge.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_merge.fit(train_vecs_merge,
                y_train_final,
                validation_data=(validation_vecs_merge, y_validation_final),
                epochs=5,
                batch_size=10,
                verbose=2)

model_merge.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/model_merge')

prediction_model_merge = np.argmax(model_merge.predict(test_vecs_merge), axis=-1)
np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/prediction_model_merge.npy', prediction_model_merge)

print(accuracy_score(prediction_model_merge, label_test))
print(classification_report(prediction_model_merge, label_test))


##########################################
# Neural Network - tfidf
train_X_tfidf  = np.load('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/train_X_tfidf.npy')
train_X_tfidf_new = train_X_tfidf[0:600000]
validation_X_tfidf_new = train_X_tfidf[600000:800000]

# Modeling of the neural nets
seed = 100
np.random.seed(seed)
model_tfidf = Sequential()
model_tfidf.add(Dense(50, activation='relu', input_dim=4984))
model_tfidf.add(Dense(6, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.0002)
model_tfidf.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model_tfidf.fit(train_X_tfidf_new,
                y_train_final,
                validation_data=(validation_X_tfidf_new, y_validation_final),
                epochs=5,
                batch_size=10,
                verbose=2)

model_tfidf.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/model_tfidf')


test_X_tfidf  = np.load('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/test_X_tfidf.npy')
prediction_model_tfidf = np.argmax(model_tfidf.predict(test_X_tfidf), axis=-1)
np.save('/Users/ganfeng/Documents/Data Science/Projects/Natural Language Processing Project/Neural_Nets/prediction_model_tfidf.npy', prediction_model_tfidf)
print(accuracy_score(prediction_model_tfidf, label_test))
print(classification_report(prediction_model_tfidf, label_test))
