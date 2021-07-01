# %%
import collections

import dateutil.parser
import pandas as pd
import tensorflow as tf

# %%
d3 = pd.read_csv(
    "firestore/data/archive (3)/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
# d3 = pd.read_csv("./data/archive (3)/cleaned_may19.csv")

d3 = d3[['id', 'reviews.date', 'asins']]
# %%
d3.info()
# d3.head()

print('sup')
# %%


def convert_to_timelines(df):
    """Convert ratings data to user."""
    timelines = collections.defaultdict(list)
    movie_counts = collections.Counter()
    for user_id, timestamp, product_id in df.values:
        timestamp = int(dateutil.parser.parse(timestamp).timestamp())
        timelines[user_id].append([product_id, timestamp])
        movie_counts[product_id] += 1

    # Sort per-user timeline by timestamp
    for (user_id, timeline) in timelines.items():
        timeline.sort(key=lambda x: x[1])
        timelines[user_id] = [movie_id for movie_id, _ in timeline]

    return timelines, movie_counts


timelines, counts = convert_to_timelines(d3)
# %%


# used to pad when user doesn't have enough context
OOV_MOVIE_ID = b''


def generate_examples_from_timelines(timelines,
                                     min_timeline_len=3,
                                     max_context_len=100):
    """Convert user timelines to tf examples.

    Convert user timelines to tf examples by adding all possible context-label
    pairs in the examples pool.

    Args:
      timelines: the user timelines to process.
      min_timeline_len: minimum length of the user timeline.
      max_context_len: maximum length of context signals.

    Returns:
      train_examples: tf example list for training.
      test_examples: tf example list for testing.
    """
    train_examples = []
    test_examples = []
    for timeline in timelines.values():
        # Skip if timeline is shorter than min_timeline_len.
        if len(timeline) < min_timeline_len:
            continue
        for label_idx in range(1, len(timeline)):
            start_idx = max(0, label_idx - max_context_len)
            context = timeline[start_idx:label_idx]

            # convert context from string to byte.
            for each_context_indx, each_context_word in enumerate(context):
                context[each_context_indx] = bytes(each_context_word, 'utf-8')

            # Pad context with out-of-vocab movie id 0.
            while len(context) < max_context_len:
                context.append(OOV_MOVIE_ID)
            label = bytes(timeline[label_idx], 'utf-8')
            feature = {
                "context":
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=context)),
                "label":
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[label]))
            }
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            if label_idx == len(timeline) - 1:
                test_examples.append(tf_example.SerializeToString())
            else:
                train_examples.append(tf_example.SerializeToString())
    return train_examples, test_examples


train_examples, test_examples = generate_examples_from_timelines(timelines)
