# In[1]:


import firebase_admin
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from firebase_admin import firestore
from numpy.testing._private.utils import build_err_msg
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from surprise import Dataset, KNNWithMeans, Reader, accuracy
from surprise.model_selection import train_test_split

# Any results you write to the current directory are saved as output.


# # Load the Dataset and Add headers
ds = pd.read_csv("../data/input/long.csv")
# %%

filtered_ds = ds[['reviews.rating', 'asins', 'reviews.username']]
filtered_ds = filtered_ds.rename(columns={
                                 'asins': 'productId', 'reviews.rating': 'Rating', 'reviews.username': 'userId'})
filtered_ds = filtered_ds.filter(['userId', 'productId', 'Rating'])
filtered_ds.head()
electronics_data = filtered_ds
filtered_ds.info()

# Getting the new dataframe which contains users who has given 50 or more ratings

new_df = electronics_data.groupby("productId").filter(
    lambda x: x['Rating'].count() >= 50)


# %%
# %%
# electronics_data.groupby(by='userId')['Rating'].count().sort_values(ascending=False).head()

# # Collaberative filtering (Item-Item recommedation)
#
# Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. We are going to use collaborative filtering (CF) approach.
# CF is based on the idea that the best recommendations come from people who have similar tastes. In other words, it uses historical item ratings of like-minded people to predict how someone would rate an item.Collaborative filtering has two sub-categories that are generally called memory based and model-based approaches.
#
#

# In[33]:


# Reading the dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df, reader)


# In[34]:


# Splitting the dataset
trainset, testset = train_test_split(data, test_size=0.3, random_state=10)


# In[35]:


# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(
    k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)


# In[36]:


# run the trained model against the testset
test_pred = algo.test(testset)


# In[37]:
user = 'Dave'

# list products
list_products = list(set(electronics_data['productId'].tolist()))
dict_item_rating = {}
for eachProduct in list_products:
    pred = algo.predict(uid=user, iid=eachProduct)
    dict_item_rating[eachProduct] = pred[3]

list_products.sort(key=lambda x: dict_item_rating[x],reverse=True)

# In[38]:


# get RMSE
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

# %%

# /*************************
# * PUSH POPULAR PORDUCTS *
# *************************/

default_app = firebase_admin.initialize_app()
db = firestore.client()

# %%
doc_ref = db.collection("rec_sys").document('user_based')

# doc_content = readFileAsDict('rec_sys2/categories.json')
doc_ref.set({user: list_products})
