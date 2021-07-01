# %%

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

# In[2]:
electronics_data = pd.read_csv(
    "../kaggle/ratings_Electronics.csv", names=['userId', 'productId', 'Rating', 'timestamp'])
electronics_data = electronics_data.iloc[:10000, 0:]

# Dropping the Timestamp column
electronics_data.drop(['timestamp'], axis=1, inplace=True)


# In[19]:
# Getting the new dataframe which contains users who has given 50 or more ratings

new_df = electronics_data.groupby("productId").filter(
    lambda x: x['Rating'].count() >= 50)


# In[39]:

ratings_matrix = new_df.pivot_table(
    values='Rating', index='userId', columns='productId', fill_value=0)


# In[41]:
# Transposing the matrix

X = ratings_matrix.T

# In[43]:

# Decomposing the Matrix
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape

# In[45]:

# Correlation Matrix

correlation_matrix = np.corrcoef(decomposed_matrix)

# %%
X.head()

# In[47]:


i = "1400501776"

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
