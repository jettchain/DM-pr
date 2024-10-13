import os
import pandas as pd

def load_and_split_data(root_dir: str) -> tuple:
    """
    Loads data from the specified directory structure and splits it into
    training and test sets based on the fold structure.
    
    Folds 1-4 are used for training and hyperparameter tuning (640 reviews),
    and Fold 5 is used as the test set (160 reviews).
    
    Returns a tuple of (train_data, test_data).
    """
    train_data = []
    test_data = []

    # Define categories for the polarity folders
    categories = {
        "negative_polarity": ["deceptive_from_MTurk", "truthful_from_Web"],
        "positive_polarity": ["deceptive_from_MTurk", "truthful_from_TripAdvisor"]
    }

    # Loop through each category and polarity folder
    for polarity, folders in categories.items():
        for folder in folders:
            label = 1 if "deceptive" in folder else 0  # Label: 1 for deceptive, 0 for truthful
            folder_path = os.path.join(root_dir, polarity, folder)

            # Separate data from folds 1-4 into train_data and fold 5 into test_data
            for fold in os.listdir(folder_path):
                fold_path = os.path.join(folder_path, fold)

                # Determine if this fold belongs to training or test set
                target_list = train_data if fold != "fold5" else test_data

                # Check if fold_path is a directory
                if os.path.isdir(fold_path):
                    for file_name in os.listdir(fold_path):
                        file_path = os.path.join(fold_path, file_name)
                        
                        # Read the text file content
                        with open(file_path, 'r') as file:
                            text = file.read()
                            target_list.append([text, label])

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data, columns=['text', 'label'])
    test_df = pd.DataFrame(test_data, columns=['text', 'label'])
    
    return train_df, test_df
