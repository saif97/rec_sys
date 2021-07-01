
# In[1]:

import json

import firebase_admin
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from firebase_admin import firestore

# %%
ds = pd.read_csv("../data/input/long.csv")
filtered_ds = ds[['reviews.rating', 'asins', 'reviews.username']]
filtered_ds = filtered_ds.rename(columns={
                                 'asins': 'productId', 'reviews.rating': 'Rating', 'reviews.username': 'userId'})
filtered_ds.head()
electronics_data = filtered_ds

# %%
ratings_mean_count = pd.DataFrame(
    electronics_data.groupby('productId')['Rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(
    electronics_data.groupby('productId')['Rating'].count())
sorted_ratings_mean_count = ratings_mean_count.sort_values(
    ['rating_counts', 'Rating'], ascending=False)
# sorted_ratings_mean_count = ratings_mean_count.sort_values(['Rating', 'rating_counts'],ascending=False)

# %%
# export into json to be uploaded
list_sorted_ratings_mean_count = sorted_ratings_mean_count.head(
    25).index.tolist()


# %%
with open('../kaggle/output/popular_products.json', 'w') as f:
    json.dump(list_sorted_ratings_mean_count, f)


# %%
# /*************************
# * PUSH POPULAR PORDUCTS *
# *************************/

default_app = firebase_admin.initialize_app()
db = firestore.client()

# %%
doc_ref = db.collection("rec_sys").document('popular_products')

# doc_content = readFileAsDict('rec_sys2/categories.json')
doc_ref.set({'all': list_sorted_ratings_mean_count})
