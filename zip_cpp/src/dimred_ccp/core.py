# Simplificado para ZIP

def to_downloadable_csv(df):
    return df.to_csv(index=False).encode('utf-8')
