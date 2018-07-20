import os
from sklearn.model_selection import train_test_split


# TODO Perhaps to handle big files you can change this, to work with the filename instead
# TODO write test.
def split_train_test(percent, dataset_file, target, dataset):
    removed_ext = os.path.splitext(dataset_file)[0]
    train_file = "{}-train.csv".format(removed_ext)
    validation_file = "{}-validation.csv".format(removed_ext)
    percent = int(percent)
    counts = dataset[target].value_counts()
    dataset = dataset[dataset[target].isin(counts[counts > 1].index)]
    target = dataset[[target]]
    test_size = (dataset.shape[0] * percent) // 100
    train_df, test_df = train_test_split(dataset, test_size=test_size, stratify=target, random_state=42)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(validation_file, index=False)
    return train_file, validation_file
