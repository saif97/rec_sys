# %%
import dateutil.parser as dp
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

ds = pd.read_csv("../data/output/long_int_id.csv")

# %%
# only extract 3 columns that I'll use in training

ds = ds[['id', 'reviews.date', 'product_id']]
ds.info()

# %%
# convert date type to epoche time stamp

test_ds = ds
# test_ds = test_ds.loc[:10]

def foo(row):
    parsed_t = dp.parse(row['reviews.date'])
    t_in_seconds = parsed_t.timestamp()
    row['reviews.date'] = str(int(t_in_seconds))
    row['product_id'] = str(row['product_id'])
    return row

test_ds= test_ds.apply(foo, axis=1)
ds = test_ds

#%%

# %%

ds.to_csv('../data/output/3_col_long_int_id.csv', index=False, header=False)
