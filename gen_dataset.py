import argparse
import pandas as pd


def process_data(data_path):
    dataframe = pd.read_csv(data_path, encoding="latin-1")

    df_new = dataframe[0:546781]
    df_new.to_csv("ner_dataset.csv", index=False)

    df_new = dataframe[0:55122]
    df_new.to_csv("ner_test_dataset.csv", index=False)

    df_new = dataframe[0:55122]
    df_new.to_csv("ner_test_quan_dataset.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_file",
        "--dataset_file",
        type=str,
        required=True,
        default=None,
        help="dataset file for training",
    )

    # Holds all the arguments passed to the function
    FLAGS = parser.parse_args()

    process_data(FLAGS.dataset_file)
