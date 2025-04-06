def thieve_confidence(df, threshold=0.5):

    return df[df['confidence'] >= threshold]