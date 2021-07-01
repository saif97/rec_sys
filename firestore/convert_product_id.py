#%%

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
dataset = pd.read_csv("../data/input/long.csv")


# %%

# /**************************************
# * REPLACE ASINS STRING TO PRODUCT ID *
# **************************************/

#%%
# rename asins to productID

dataset.shape

#%%
# remove reviews_dateSeen column since it causes issues in big query.
dataset = dataset.drop(['reviews.dateSeen'],axis=1)
dataset.info()


#%%
# write the updated data frame to csv file.
dataset.to_csv('../data/output/long_int_id.csv')