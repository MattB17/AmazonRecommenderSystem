import json
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime



print("----------------------")
print("LOADING DATA")
print("----------------------")

data = None
with open(os.path.join('data', 'train.json'), 'r') as train_file:
    data = [json.loads(row) for row in train_file]


data_df = pd.DataFrame(data).drop(columns=['image'])
del data

test_data = None
with open(os.path.join('data', 'test.json'), 'r') as test_file:
    test_data = [json.loads(row) for row in test_file]


test_data_df = pd.DataFrame(test_data)
del test_data

print("----------------------")
print("PREPROCESSING")
print("----------------------")


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


def preprocess(data_df, genres):
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

    return data_df


music_categories = data_df['category'].unique()

data_df = preprocess(data_df, music_categories)
test_data_df = preprocess(test_data_df, music_categories)


def calculate_MSE(actuals, predicteds):
    """Calculates the Mean Squared Error between `actuals` and `predicteds`.

    Parameters
    ----------
    actuals: np.array
        A numpy array of the actual values.
    predicteds: np.array
        A numpy array of the predicted values.

    Returns
    -------
    float
        A float representing the Mean Squared Error between `actuals` and
        `predicteds`.

    """
    return (((actuals - predicteds)**2).sum()) / (len(actuals))


X_train = data_df.drop(columns=['overall'])
y_train = data_df['overall']
X_test = test_data_df

print("----------------------")
print("COLLABORATIVE FILTERING")
print("----------------------")


def user_item_matrix(df, rating_col, user_col, item_col):
    return sp.csr_matrix(df[rating_col], (df[user_col], df[item_col]))


train_data = data_df

categories = train_data['category'].unique()
dfs = []
for category in categories:
    dfs.append(train_data[train_data['category'] == category].sample(frac=0.25))
train_data = pd.concat(dfs, axis=0)
train_data = data_df.sort_index()

train_data['itemID'] = train_data['itemID'].astype("category")
train_data['reviewerID'] = train_data['reviewerID'].astype("category")


import scipy.sparse as sp

item_matrix = train_data.pivot(index='itemID', columns='reviewerID', values='overall')
item_matrix = item_matrix.fillna(0)
user_item_train_matrix = sp.csr_matrix(item_matrix.values)

print("----------------------")
print("CALCULATING AVERAGE")
print("----------------------")

global_average = train_data['overall'].mean()


from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(user_item_train_matrix)
item_neighbors = np.asarray(model_knn.kneighbors(user_item_train_matrix, return_distance=False))

print("--------------------------")
print("FIT FIRST KNN")
print("--------------------------")


user_matrix = train_data.pivot(index='reviewerID', columns='itemID', values='overall')
user_matrix = user_matrix.fillna(0)
user_item_train_matrix = sp.csr_matrix(user_matrix.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(user_item_train_matrix)
user_neighbors = np.asarray(model_knn.kneighbors(user_item_train_matrix, return_distance=False))

print("--------------------------------")
print("FIT SECOND KNN")
print("--------------------------------")


train_user_avg = train_data.groupby(train_data['reviewerID'], as_index=False)['overall'].mean()
train_item_avg = train_data.groupby(train_data['itemID'], as_index=False)['overall'].mean()
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
    df_final = df_final.fillna(total_avg)
    df_final.index = X.index
    return df_final


X_test_aug = weighted_average_data(X_test, global_average, user_avgs, item_avgs)
X_test_mod = pd.concat([X_test, X_test_aug], axis=1)


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
X_test_mod['pred'] = X_test_mod['pred'].apply(lambda x: threshold_rating(x))


print("----------------------")
print("LANGUAGE MODEL")
print("----------------------")


columns_to_keep = ['cleanedPrice', 'isPop', 'isAlternativeRock', 'isJazz', 'isClassical', 'isDanceElectronic', 'reviewWordCount']
X_train_reg1 = X_train[columns_to_keep]
X_test_reg1 = X_test[columns_to_keep]


from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
X_train_reg1['reviewWordCount'] = X_train_reg1['reviewWordCount'].apply(lambda x: np.log(x))
X_test_reg1['reviewWordCount'] = X_test_reg1['reviewWordCount'].apply(lambda x: np.log(x))


def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

X_train_reg1 = clean_dataset(X_train_reg1)
y_train1 = y_train[y_train.index.isin(X_train_reg1.index)]
X_train1 = X_train[X_train.index.isin(X_train_reg1.index)]

X_test_reg1 = clean_dataset(X_test_reg1)
X_test1 = X_test[X_test.index.isin(X_test_reg1.index)]


X_train_reg1 = min_max_scaler.fit_transform(X_train_reg1)
X_test_reg1 = min_max_scaler.transform(X_test_reg1)


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
X_train1['processedReview'] = X_train1['fullReviewText'].apply(lambda x: process_review_text(x, exclude_english, ps))
X_test1['processedReview'] = X_test1['fullReviewText'].apply(lambda x: process_review_text(x, exclude_english, ps))


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X_train_cv1 = cv.fit_transform(X_train1['processedReview'])
X_test_cv1 = cv.transform(X_test1['processedReview'])

import scipy.sparse as sp

X_train_reg1_sp = sp.csr_matrix(X_train_reg1)
X_train_cv_reg1 = sp.hstack((X_train_cv1, X_train_reg1_sp), format='csr')

X_test_reg1_sp = sp.csr_matrix(X_test_reg1)
X_test_cv_reg1 = sp.hstack((X_test_cv1, X_test_reg1_sp), format='csr')


from xgboost import XGBRegressor
xg_reg = XGBRegressor(learning_rate=0.3, n_estimators=500, max_depth=2)
xg_reg.fit(X_train_cv_reg1, y_train1)

predictions = xg_reg.predict(X_test_cv_reg1)


preds = pd.DataFrame(predictions, columns=['prediction'])
X_test['userID-itemID'] = X_test['reviewerID'] + "-" + X_test['itemID']

preds = preds.reset_index()[['prediction']]
ids = X_test[['userID-itemID']].reset_index()[['userID-itemID']]
final_preds_reg = pd.concat([ids, preds], axis=1)
final_preds_reg.index = final_preds_reg['userID-itemID']


print("----------------------")
print("FINAL PREDICTIONS")
print("----------------------")


predictions = open(os.path.join('data', 'rating_predictions.csv'), 'w')
for l in open(os.path.join('data', 'rating_pairs.csv')):
    if l.startswith('userID'):
        predictions.write(l)
        continue
    id = l.strip()
    if id in final_preds_reg.index:
        predictions.write(id+ ',' + str(final_preds_reg['prediction'][id]) + '\n')
    elif id in X_test_mod.index:
        predictions.write(id + ',' + str(X_test_mod['pred'][id]) + '\n')
    else:
        predictions.write(id + ',' + str(global_average) + '\n')
