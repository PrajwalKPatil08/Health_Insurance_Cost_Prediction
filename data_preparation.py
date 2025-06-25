import pandas as pd

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Encode categorical columns
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'yes': 0, 'no': 1})
    region_map = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
    df['region'] = df['region'].map(region_map)

    X = df.drop("charges", axis=1)
    Y = df["charges"]
    return X, Y
