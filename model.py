import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import scipy.sparse as sp
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import CountVectorizer


###############################
# LOAD DATA
###############################
data = None
with open(os.path.join('data', 'train.json'), 'r') as train_file:
  data = [json.loads(row) for row in train_file]
data_df = pd.DataFrame(data)
del data

data_df_cf = data_df[['reviewerID', 'itemID', 'overall']]


###############################
# PREPROCESSING
###############################
def trim_price(price):
    """Trims `price` to remove the $ sign.

    If the price variable does not have the format $x.xx
    then the empty string is returned.

    Parameters
    ----------
    price: str
        A string representing a price.

    Returns
    -------
    str
        A string representing `price` but with the $ sign removed,
        or the empty string if `price` does not have the correct
        format.

    """
    if (not pd.isnull(price) and isinstance(price, str) and
        len(price) > 0 and price[0] == '$'):
        return price[1:]
    return ""


def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


data_df['reviewMonth'] = data_df['reviewTime'].apply(lambda x: x.split(' ')[0])
data_df['reviewYear'] = data_df['reviewTime'].apply(lambda x: x.split(' ')[2])
data_df['reviewHour'] = data_df['unixReviewTime'].apply(lambda x: datetime.fromtimestamp(x).hour)
data_df['reviewMonthYear'] = data_df['reviewYear'] + '-' + data_df['reviewMonth']

data_df['cleanedPrice'] = data_df['price'].apply(lambda x: trim_price(x))
data_df = data_df[data_df['cleanedPrice'] != ""]
data_df['cleanedPrice'] = data_df['cleanedPrice'].astype('float')

data_df['fixedReviewText'] = np.where(pd.isnull(data_df['reviewText']), "", data_df['reviewText'])
data_df['fixedSummary'] = np.where(pd.isnull(data_df['summary']), "", data_df['summary'])
data_df['fullReviewText'] = data_df['fixedSummary'] + " " + data_df['fixedReviewText']

data_df = data_df.drop(columns=['fixedReviewText', 'fixedSummary'])

genres = data_df['category'].unique()

for genre in genres:
    genre_col = "is" + genre.replace(" ", "").replace("&", "")
    data_df[genre_col] = data_df['category'].apply(lambda x: 1 if x == genre else 0)

data_df['reviewWordCount'] = data_df['fullReviewText'].apply(lambda x: len(x.split()))
X_train = data_df

###############################
# TRAINING
###############################
columns_to_keep = ['cleanedPrice', 'isPop', 'isAlternativeRock', 'isJazz', 'isClassical', 'isDanceElectronic', 'reviewWordCount']
X_train_reg = data_df[columns_to_keep]

min_max_scaler = MinMaxScaler()

X_train_reg['reviewWordCount'] = X_train_reg['reviewWordCount'].apply(lambda x: 0 if pd.isnull(x) or not np.isfinite(x) else np.log(x))
X_train_reg = clean_dataset(X_train_reg)

X_train = X_train[X_train.index.isin(X_train_reg.index)]
y_train = X_train['overall']

print("Sliced")

X_train_reg = min_max_scaler.fit_transform(X_train_reg)

def process_review_text(review_text, exclude_text, ps):
    """Pre-processes the text given by `review_text`.

    Parameters
    ----------
    review_text: str
        The review text to be processed.
    exclude_text: collection
        A collection of words to be excluded.
    ps: PorterStemmer
        The PorterStemmer used to perform word stemming.

    Returns
    -------
    str
        A string representing the processed version of `review_text`.

    """
    review = re.sub('[^a-zA-Z0-9]', ' ', review_text).lower().split()
    review = [ps.stem(word) for word in review if not word in exclude_text]
    return ' '.join(review)


exclude_english = set(stopwords.words('english'))
ps = PorterStemmer()
X_train['processedReview'] = X_train['fullReviewText'].apply(lambda x: process_review_text(x, exclude_english, ps))

tfidf = CountVectorizer(max_features=1500)
X_train_tfidf = tfidf.fit_transform(X_train['processedReview'])

X_train_reg_sp = sp.csr_matrix(X_train_reg)
X_train_tfidf_reg = sp.hstack((X_train_tfidf, X_train_reg_sp), format='csr')

print("Got Matrix")

reg_model = XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=2)
reg_model.fit(X_train_tfidf_reg, y_train)

average_rating = y_train.mean()

###############################
# PREDICTION
###############################
data = None
with open(os.path.join('data', 'test.json'), 'r') as test_file:
  data = [json.loads(row) for row in test_file]
data_df = pd.DataFrame(data)
del data

test_df_cf = data_df[['reviewerID', 'itemID', 'overall']]

data_df['reviewMonth'] = data_df['reviewTime'].apply(lambda x: x.split(' ')[0])
data_df['reviewYear'] = data_df['reviewTime'].apply(lambda x: x.split(' ')[2])
data_df['reviewHour'] = data_df['unixReviewTime'].apply(lambda x: datetime.fromtimestamp(x).hour)
data_df['reviewMonthYear'] = data_df['reviewYear'] + '-' + data_df['reviewMonth']

data_df['cleanedPrice'] = data_df['price'].apply(lambda x: trim_price(x))
data_df = data_df[data_df['cleanedPrice'] != ""]
data_df['cleanedPrice'] = data_df['cleanedPrice'].astype('float')

data_df['fixedReviewText'] = np.where(pd.isnull(data_df['reviewText']), "", data_df['reviewText'])
data_df['fixedSummary'] = np.where(pd.isnull(data_df['summary']), "", data_df['summary'])
data_df['fullReviewText'] = data_df['fixedSummary'] + " " + data_df['fixedReviewText']

data_df = data_df.drop(columns=['fixedReviewText', 'fixedSummary'])

for genre in genres:
    genre_col = "is" + genre.replace(" ", "").replace("&", "")
    data_df[genre_col] = data_df['category'].apply(lambda x: 1 if x == genre else 0)

data_df['reviewWordCount'] = data_df['fullReviewText'].apply(lambda x: len(x.split()))
X_test = data_df

columns_to_keep = ['cleanedPrice', 'isPop', 'isAlternativeRock', 'isJazz', 'isClassical', 'isDanceElectronic', 'reviewWordCount']
X_test_reg = X_test[columns_to_keep]

X_test_reg['reviewWordCount'] = X_test_reg['reviewWordCount'].apply(lambda x: 0 if pd.isnull(x) or not np.isfinite(x) else np.log(x))

X_test_reg = clean_dataset(X_test_reg)

X_test = X_test[X_test.index.isin(X_test_reg.index)]

print("Sliced")

X_test_reg = min_max_scaler.transform(X_test_reg)

X_test['processedReview'] = X_test['fullReviewText'].apply(lambda x: process_review_text(x, exclude_english, ps))

X_test_tfidf = tfidf.transform(X_test['processedReview'])

X_test_reg_sp = sp.csr_matrix(X_test_reg)
X_test_tfidf_reg = sp.hstack((X_test_tfidf, X_test_reg_sp), format='csr')

predictions = reg_model.predict(X_test_tfidf_reg)

preds = pd.DataFrame(predictions, columns=['prediction'])
X_test['userID-itemID'] = X_test['reviewerID'] + "-" + X_test['itemID']

final_preds_reg = pd.concat([X_test[['userID-itemID']], preds], axis=1)
final_preds_reg.index = final_preds['userID-itemID']

################################
# Collaborative Filtering
################################
def user_item_matrix(df, rating_col, user_col, item_col):
    return sp.csr_matrix(df[rating_col], (df[user_col], df[item_col]))

item_matrix = data_df_cf.pivot(index='itemID', columns='reviewerID', values='overall')
item_matrix = item_matrix.fillna(0)
user_item_train_matrix = sp.csr_matrix(item_matrix.values)

global_average = data_df_cf['overall'].mean()

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(user_item_train_matrix)
item_neighbors = np.asarray(model_knn.kneighbors(user_item_train_matrix, return_distance=False))

user_matrix = data_df_cf.pivot(index='reviewerID', columns='itemID', values='overall')
user_matrix = user_matrix.fillna(0)
user_item_train_matrix = sp.csr_matrix(user_matrix.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(user_item_train_matrix)
user_neighbors = np.asarray(model_knn.kneighbors(user_item_train_matrix, return_distance=False))

train_user_avg = data_df_cf.groupby(train_data['reviewerID'], as_index=False)['overall'].mean()
train_item_avg = data_df_cf.groupby(train_data['itemID'], as_index=False)['overall'].mean()
train_user_avg.columns = ['reviewerID', 'userAverage']
train_item_avg.columns = ['itemID', 'itemAverage']
train_user_avg = train_user_avg.set_index('reviewerID')
train_item_avg = train_item_avg.set_index('itemID')

item_avgs = []
for i in range(len(item_neighbors)):
    item_avgs.append(train_item_avg['itemAverage'][item_matrix.index[item_neighbors[i]]].mean())

item_avgs = pd.concat([pd.DataFrame(item_matrix.index, columns=['itemID']), pd.DataFrame(item_avgs, columns=['itemRating'])], axis=1)

user_avgs = []
for i in range(len(user_neighbors)):
    user_avgs.append(train_user_avg['userAverage'][user_matrix.index[user_neighbors[i]]].mean())

user_avgs = pd.concat([pd.DataFrame(user_matrix.index, columns=['reviewerID']), pd.DataFrame(user_avgs, columns=['userRating'])], axis=1)

def weighted_average_data(X, total_avg, user_avgs, item_avgs):
    """Calculates the error based on the weighted average prediction.

    Parameters
    ----------
    X: pd.DataFrame
        The DataFrame of features.
    y: np.array
        A numpy array containing the targets
    total_avg: float
        The average across all users/items.
    user_avgs: pd.DataFrame
        A DataFrame containing the average rating for each user.
    item_avgs: pd.DataFrame
        A DataFrame containing the average rating for each item.

    Returns
    -------
    float
        A float representing the mean squared error of the predictions.

    """
    df_user = pd.merge(X, user_avgs, how='left', on=['reviewerID'])
    df_final = pd.merge(df_user, item_avgs, how='left', on=['itemID'])
    df_final = df_final[['userRating', 'itemRating']]
    df_final.fillna(total_avg)
    return df_final

X_test_aug = weighted_average_data(test_df_cf, global_average, user_avgs, item_avgs)
X_test_mod = pd.merge(test_df_cf, X_test_aug, how='left', left_index=True, right_index=True)

def threshold_rating(rating):
    """Thresholds `rating` to lie in the range [1, 5].

    Parameters
    ----------
    rating: float
        The rating to be thresholded.

    Returns
    -------
    float
        A float representing the thresholded rating.

    """
    if rating < 1:
        return 1
    if rating > 5:
        return 5
    return rating

X_test_mod['pred'] = (0.5 * X_test_mod['userRating']) + (0.5 * X_test_mod['itemRating'])
X_test_mod['pred'].apply(lambda x: threshold_rating(x))

X_test_mod['userID-itemID'] = X_test_mod['reviewerID'] + "-" + X_test_mod['itemID']
X_test_mod.index = X_test_mod['userID-itemID']

###############################
# Outputting
###############################

predictions = open(os.path.join('data', 'rating_predictions.csv'), 'w')
for l in open(os.path.join('data', 'rating_pairs.csv')):
    if l.startswith('userID'):
        predictions.write(l)
        continue
    id = l.strip()
    if id in final_preds.index:
        predictions.write(id+ ',' + str(final_preds['prediction'][id]) + '\n')
    elif id in X_test_mod.index:
        predictions.write(id + ',' + str(X_test_mod['pred'][id]) + '\n')
    else:
        predictions.write(id + ',' + str(global_average) + '\n')
