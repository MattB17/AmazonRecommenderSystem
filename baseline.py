from collections import defaultdict
import json
import os

allRatings = []
userRatings = defaultdict(list)

with open(os.path.join('data', 'train.json'), 'r') as train_file:
    for row in train_file:
        data = json.loads(row)
        r = float(data['overall'])
        allRatings.append(r)
        userRatings[data['reviewerID']].append(r)

globalAverage = sum(allRatings)/len(allRatings)
userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open(os.path.join('data', 'rating_predictions.csv'), 'w')
for l in open(os.path.join('data', 'rating_pairs.csv')):
    if l.startswith('userID'):
        #header
        predictions.write(l)
        continue
    u,p = l.strip().split('-')
    if u in userAverage:
        predictions.write(u + '-' + p + ',' + str(userAverage[u]) + '\n')
    else:
        predictions.write(u + '-' + p + ',' + str(globalAverage) + '\n')
