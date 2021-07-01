# In[1]:

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from surprise import Dataset, KNNWithMeans, Reader, accuracy
from surprise.model_selection import train_test_split

InteractiveShell.ast_node_interactivity = "all"
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.neighbors import NearestNeighbors
# from sklearn.externals import joblib
# import scipy.sparse
# from scipy.sparse import csr_matrix
warnings.simplefilter('ignore')
# get_ipython().run_line_magic('matplotlib', 'inline')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Load the Dataset and Add headers

# In[2]:
electronics_data = pd.read_csv(
    "../kaggle/ratings_Electronics.csv", names=['userId', 'productId', 'Rating', 'timestamp'])
electronics_data = electronics_data.iloc[:1048576, 0:]

# Dropping the Timestamp column
electronics_data.drop(['timestamp'], axis=1, inplace=True)


# In[14]:
# Analysis of rating given by the user

no_of_rated_products_per_user = electronics_data.groupby(
    by='userId')['Rating'].count().sort_values(ascending=False)
# In[19]:
# Getting the new dataframe which contains users who has given 50 or more ratings

new_df = electronics_data.groupby("productId").filter(
    lambda x: x['Rating'].count() >= 50)

new_df.head()

# In[20]:

no_of_ratings_per_product = new_df.groupby(
    by='productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('No of ratings per product')
ax.set_xticklabels([])

plt.show()

# In[22]:
# Average rating of the product
new_df.groupby('productId')['Rating'].mean(
).sort_values(ascending=False).head()

# In[23]:
# Total no of rating for product

new_df.groupby('productId')['Rating'].count(
).sort_values(ascending=False).head()

# In[24]:
ratings_mean_count = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())

# In[25]:
ratings_mean_count['rating_counts'] = pd.DataFrame(
    new_df.groupby('productId')['Rating'].count())

# In[26]:
ratings_mean_count.head()

# In[27]:

ratings_mean_count['rating_counts'].max()

# In[28]:


plt.figure(figsize=(8, 6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)


# In[29]:


plt.figure(figsize=(8, 6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Rating'].hist(bins=50)


# In[30]:

plt.figure(figsize=(8, 6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='rating_counts',
              data=ratings_mean_count, alpha=0.4)


# In[31]:


popular_products = pd.DataFrame(new_df.groupby('productId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(30).plot(kind="bar")


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


test_pred


# In[38]:


# get RMSE
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)


# # Model-based collaborative filtering system
#
# These methods are based on machine learning and data mining techniques. The goal is to train models to be able to make predictions. For example, we could use existing user-item interactions to train a model to predict the top-5 items that a user might like the most. One advantage of these methods is that they are able to recommend a larger number of items to a larger number of users, compared to other methods like memory based approach. They have large coverage, even when working with large sparse matrices.

# In[39]:


new_df1 = new_df.head(10000)
ratings_matrix = new_df1.pivot_table(
    values='Rating', index='userId', columns='productId', fill_value=0)
ratings_matrix.head()


# As expected, the utility matrix obtaned above is sparce, I have filled up the unknown values wth 0.
#
#

# In[40]:


ratings_matrix.shape


# Transposing the matrix

# In[41]:


X = ratings_matrix.T
X.head()


# In[42]:


X.shape


# Unique products in subset of data
#

# In[43]:


X1 = X


# In[44]:


# Decomposing the Matrix
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape


# In[45]:


# Correlation Matrix

correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# In[46]:


X.index[75]


# Index # of product ID purchased by customer
#
#

# In[47]:


i = "B00000K135"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID


# Correlation for all items with the item purchased by this customer based on items rated by other customers people who bought the same product

# In[48]:


correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape


# Recommending top 25 highly correlated products in sequence
#
#

# In[49]:


Recommend = list(X.index[correlation_product_ID > 0.65])

# Removes the item already bought by the customer
Recommend.remove(i)

Recommend[0:24]


# Here are the top 10 products to be displayed by the recommendation system to the above customer based on the purchase history of other customers in the website.
#
