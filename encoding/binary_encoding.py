from category_encoders import BinaryEncoder


def binary_encode(df, cols_to_encode):
    print("Binary encode cols", cols_to_encode)
    encoder = BinaryEncoder(cols=cols_to_encode, return_df=True)
    data_encoded = encoder.fit_transform(df)
    return data_encoded
