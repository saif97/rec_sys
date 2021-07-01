# %%
# It's recommended to use vs code to run the python files. It'll read the # %% symbols as cell which can be executed individually.
import json

import firebase_admin
from firebase_admin import credentials, firestore


def readFileAsDict(path):
    # Opening JSON file
    with open(path) as json_file:
        return json.load(json_file)


# In a production application, Admin SDK keys shouldn't be shared, added to a repo, or hardcoded into a project. Additionally, the path to the file should be added as env variables.
# So that the SDK can access it. That being said, I've included the file in the project so you can run it on your machine.
# simply add this to your bash/zsh rc export GOOGLE_APPLICATION_CREDENTIALS="path to file"
# if you want me to add you into the firebase project let me know.
default_app = firebase_admin.initialize_app()
db = firestore.client()

# %%
# /**********************
#  * UPLOAD CATEGORIES *
#  **********************/
doc_ref = db.collection('test_dataset').document('categories')
doc_content = readFileAsDict('../test_categories.json')
# doc_content = readFileAsDict('rec_sys2/categories.json')
doc_ref.set(doc_content)

# %%
# /*****************
# * PUSH PRODUCTS *
# *****************/
doc_ref = db.collection('products2')
listProducts = readFileAsDict('../data/output/products.json')
# listProducts = readFileAsDict('products.json')

for i, v in enumerate(listProducts):
    doc_ref.document(v).set(listProducts[v])


# %%

# /*************************
# * PUSH POPULAR PORDUCTS *
# *************************/
