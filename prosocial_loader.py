import pandas as pd
import ast

def preprocess_data(df):
    df = df.drop(columns=["etc", "dialogue_id", "response_id", "episode_done"])
    # Initializing dictionaries to store datasets for each safety annotation
    datasets = {'casual': [], 'needs caution': [], 'needs intervention': []}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        annotations = ast.literal_eval(row['safety_annotations'])

        # Iterate over each safety annotation for the current row
        for annotation in annotations:
            # Determining the safety annotation category
            category = annotation.lower()

            # Creating a copy of the row with only the corresponding safety annotation
            new_row = row.copy()
            new_row['safety_annotations'] = [annotation]

            # Adding the row to the corresponding dataset
            datasets[category].append(new_row)

    return datasets


def load_prosocial_dialog_dataset():
    # Load the dataset
    df = pd.read_csv("prosocial_dataset.csv")

    # Combine 'context' and 'response' columns into a single 'query' column
    df['query'] = df['context'] + ' ' + df['response']

    # Drop 'context' and 'response' columns
    df = df.drop(columns=['context', 'response'])

    # Reorder columns
    cols = ['query'] + [col for col in df if col != 'query']
    df = df[cols]

    df['safety_label'] = df['safety_label'].str.replace('_', '').str.strip()

    return df

def save_datasets_to_csv(datasets):
    for category, dataset in datasets.items():
        # to convert list of rows to DataFrame
        dataset = pd.DataFrame(dataset)
        # to convert 'safety_annotations' column to string
        dataset['safety_annotations'] = dataset['safety_annotations'].apply(lambda x: x[0])
        # to remove duplicate rows
        dataset = dataset.drop_duplicates()
        filename = f"{category}_annotations.csv"
        dataset.to_csv(filename, index=False)
        # print(f"Saved {filename}")

if __name__ == "__main__":
    # Load the dataset
    df = load_prosocial_dialog_dataset()

    # Preprocess the dataset
    preprocessed_datasets = preprocess_data(df)

    # Saving preprocessed datasets to CSV files
    save_datasets_to_csv(preprocessed_datasets)


