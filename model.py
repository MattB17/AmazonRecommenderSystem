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
from sklearn.feature_extraction.text import TfidfVectorizer


###############################
# LOAD DATA
###############################
data = None
with open(os.path.join('data', 'train.json'), 'r') as train_file:
  data = [json.loads(row) for row in train_file]
data_df = pd.DataFrame(data)
del data

data_df_cf = data_df[['reviewerID', 'itemID', 'overall']]

global_average = data_df['overall'].mean()


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

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train['processedReview'])

X_train_reg_sp = sp.csr_matrix(X_train_reg)
X_train_tfidf_reg = sp.hstack((X_train_tfidf, X_train_reg_sp), format='csr')

print("Got Matrix")

reg_model = XGBRegressor(learning_rate=0.3, n_estimators=5000, max_depth=1)
reg_model.fit(X_train_tfidf_reg, y_train)

###############################
# PREDICTION
###############################
data = None
with open(os.path.join('data', 'test.json'), 'r') as test_file:
  data = [json.loads(row) for row in test_file]
data_df = pd.DataFrame(data)
del data

test_df_cf = data_df[['reviewerID', 'itemID']]

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

preds = preds.reset_index()[['prediction']]
ids = X_test[['userID-itemID']].reset_index()[['userID-itemID']]
final_preds_reg = pd.concat([ids, preds], axis=1)
final_preds_reg.index = final_preds_reg['userID-itemID']

def threshold_rating(rating):
    if rating < 1:
        return 1
    if rating > 5:
        return 5
    return rating

final_preds_reg['prediction'] = final_preds_reg['prediction'].apply(lambda x: threshold_rating(x))

###############################
# Outputting
###############################

predictions = open(os.path.join('data', 'rating_predictions.csv'), 'w')
for l in open(os.path.join('data', 'rating_pairs.csv')):
    if l.startswith('userID'):
        predictions.write(l)
        continue
    id = l.strip()
    if id in final_preds_reg.index:
        predictions.write(id+ ',' + str(final_preds_reg['prediction'][id]) + '\n')
    else:
        predictions.write(id + ',' + str(global_average) + '\n')
