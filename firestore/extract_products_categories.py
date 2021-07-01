
# this file is responsible for cleaning the data I got from the link below.
# https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products
# link to the file used: https://drive.google.com/file/d/1HbxqySJzSOTOQmnGtO1CeFPnpFouDgUP/view?usp=sharing
# after cleaning the DataFrame I store it in json then upload it to firebase using the firebase.py file.

# It's recommended to use vs code to run the python files. It'll read the # %% symbols as cell which can be executed individually.
# %%
import json
import uuid

import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)

# %%
# for some reason, jupyter notebook is finicky sometimes is reads the file sometime it dosen't.
# if you get <FileNotFoundError: [Errno 2] No such file or directory:> errors try setting the file path
# to relative to this file.
d3 = pd.read_csv("../data/output/long_int_id.csv")
# d3 = pd.read_csv(
# "firestore/data/archive (3)/cleaned_may19.csv")

products = d3
products.head()
# %%
# todo: move to new file
# /**********************
#  * EXTRACT CATEGORIES *
#  **********************/
# parse product categorise.
# some products have multiple categorise assigned to them (similar to tages) thus I had to concatenate them into a single string then split them.
dataSet_categories = d3
categories = set(dataSet_categories.categories)
concat_categories = ''
for i in categories:
    concat_categories += (',' + i)
# the standard set function will remove duplicated elements
categories = set(concat_categories.split(','))
print(categories)

# %%
# This cell is responsible for creating unique indexes for each category which then will be used in firebase as index for each category & products themselves.
cat2Hash = {}
hash2Cat = {}

for cat in categories:
    hash = uuid.uuid4().hex
    cat2Hash[cat] = hash
    # remove the extra white space around the categories.
    hash2Cat[hash] = cat.strip()

# %%
# save category dict to json file. which then will be uploaded to firebase in firebase.py
with open("test_categories.json", 'w') as fp:
    # with open("categories.json", 'w') as fp:
    json.dump(hash2Cat, fp)


# %%
# /*********************************
# * EXTRACT AVERAGE STARS & COUNT *
# *********************************/
product_avg_review = d3[['asins', 'reviews.rating']].groupby('asins').mean()
product_count_review = d3[['asins', 'reviews.rating']].groupby('asins').count()
product_count_review.head()

# %%
# /********************
#  * EXTRACT PRODUCTS *
#  ********************/

# only select the information that's used in the app.
dataSet_products = d3[['asins', 'name', 'brand',  'categories',
                       'primaryCategories', 'imageURLs']]

dataSet_products.head()

# The dataset contrins reviews for products. This statement gets rid of the dublicates.
products = dataSet_products.drop_duplicates('asins')
products.head(60)

# %%
dictJson = {}
for i, eachProduct in products.iterrows():
    # convert each row dictionary into a json string
    s = eachProduct.to_json()

    # convert the json string into a json object.
    outJson = json.loads(s)

    # Split categories into a list.
    cats = outJson['categories']
    cats = cats.split(',')
    # get rid of the white spaces.
    cats = [eachCat.strip() for eachCat in cats]
    outJson['categories'] = cats

    # split urls
    urls = outJson['imageURLs']
    outJson['imageURLs'] = urls.split(',')

    # add hash ids to each product.
    # hash = uuid.uuid4().hex
    product_id = outJson['asins']
    outJson.pop('asins', None)
    outJson['review_avg'] = round(product_avg_review.loc[product_id][0], 1)
    outJson['review_count'] = int(product_count_review.loc[product_id][0])

    dictJson[product_id] = outJson

# The uuids are similar to keys in a hash table which will allow firebase to have constant time lookup.

with open("../data/output/products.json", 'w') as fp:
    # with open("products.json", 'w') as fp:
    json.dump(dictJson, fp)
