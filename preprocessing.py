from imports import *

def returnDatasetInfo(df):

    # Return the basic structure info about the dataset
    print("Shape \n{}\n\n".format(df.shape))
    print("Info \n{}\n\n".format(df.info()))
    print("Description \n{}\n\n".format(df.describe()))
    print("Missing values check \n{}\n\n".format(df.isnull().any()))

def preprocessData(df):

    # Remove unnecessary stuff
    for x in df['text']:
        x = re.sub('[^a-zA-Z]', ' ', x)
    for x in df['aspect_term']:
        x = re.sub('[^a-zA-Z]', ' ', x)

    # Make all the capital letters small
    df['text'] = df['text'].str.lower()
    df['aspect_term'] = df['aspect_term'].str.lower()

    # Remove [comma] from the column df[' text']
    df['text'] = df['text'].replace("comma", "", regex=True)
    df['text'] = df['text'].replace("\[]", "", regex=True)

    # Remove [comma] from the column df['aspect_term']
    df['aspect_term'] = df['aspect_term'].replace("comma", "", regex=True)
    df['aspect_term'] = df['aspect_term'].replace("\[]", "", regex=True)

    # Remove _ from the text
    df['text'] = df['text'].replace('_', '', regex=True)
    df['aspect_term'] = df['aspect_term'].replace('_', '', regex=True)

    # Remove special characters from text
    df['text'] = df['text'].apply(lambda x: re.sub('\W+', ' ', x))
    df['aspect_term'] = df['aspect_term'].apply(lambda x: re.sub('\W+', ' ', x))

    # Remove the stop words
    # nltk.download()
    stopWords = set(stopwords.words("english"))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in (stopWords)]))

    # Tag the words
    # Each word is tagged with its type eg. Adjective, Noun, etc
    # Chunk them together and return
    def tagWords(sentence):
        words = word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        return tagged

    df['tagged_words'] = df['text'].apply(lambda row: tagWords(row))

    return df
