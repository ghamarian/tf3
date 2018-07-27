import os
import itertools
from sklearn.model_selection import train_test_split


# TODO Perhaps to handle big files you can change this, to work with the filename instead
# TODO write test.
def split_train_test(percent, dataset_file, target, dataframe):
    removed_ext = os.path.splitext(dataset_file)[0]
    train_file = "{}-train.csv".format(removed_ext)
    validation_file = "{}-validation.csv".format(removed_ext)
    percent = int(percent)
    counts = dataframe[target].value_counts()
    dataframe = dataframe[dataframe[target].isin(counts[counts > 1].index)]
    target = dataframe[[target]]
    test_size = (dataframe.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataframe, test_size=test_size, stratify=target, random_state=42)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(validation_file, index=False)
    return train_file, validation_file


def insert_data(df, categories, unique_values, default_list, frequent_values2frequency, SAMPLE_DATA_SIZE):
    data = df.head(SAMPLE_DATA_SIZE).T
    data.insert(0, 'Defaults', default_list.values())
    data.insert(0, '(most frequent, frequency)', frequent_values2frequency.values())
    data.insert(0, 'Unique Values', unique_values)
    data.insert(0, 'Category', categories)
    sample_column_names = ["Sample {}".format(i) for i in range(1, SAMPLE_DATA_SIZE + 1)]
    data.columns = list(
        itertools.chain(['Category', '#Unique Values', '(Most frequent, Frequency)', 'Defaults'],
                        sample_column_names))
    return data
