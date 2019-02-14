# #### Data Pre-processing
# - tokenization :
#   - split the text into sentences and then sentences into words
#   - lower case words
#   - remove punctuation
#   - remove words with fewer than 2 characters
#   - remove all stopwords + added ('table', 'legend')
# - Lemmatization :
#  - verbs in past and future tenses are changed into present
# - Stemmazation :
#  - words are reduced to their root form
class Preprocessing_util:

    import gensim
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    from gensim.parsing.preprocessing import strip_numeric
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    # from nltk.stem.porter import *
    import nltk
    import spacy
    import numpy as np

    def __init__(self, text):
        self.bigram_mod, self.trigram_mod = init_n_gram_mods(text)

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(text):
        result = []
    #     deacc=True removes punctuations
        for token in gensim.utils.simple_preprocess(text, deacc=True):
            if token not in STOP_WORDS and len(token)>1:
    #             result.append(lemmatize_stemming(strip_numeric(token)))
                result.append(lemmatize_stemming(token))
        return result

    def make_bigram(text):
        processed_text = [[word for word in simple_preprocess(str(doc)) if word not in STOP_WORDS] for doc in text]
        return [bigram_mod[doc] for doc in processed_text]

    def make_trigram(text):
        processed_text = [[word for word in simple_preprocess(str(doc)) if word not in STOP_WORDS] for doc in text]
        return trigram_mod[[bigram_mod[doc] for doc in processed_text]]

    def explain_make_trigram(text):
        conversion = {}
        for doc in text:
            pro_doc = simple_preprocess(str(doc))
            if doc!= pro_doc:
                print(doc)
                print(pro_doc)
                conversion[doc]=pro_doc
        return conversion

    def n_gram_lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        text_out = []
        for sent in text:
            doc = nlp(' '.join(sent))
            text_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return text_out

    def explain_n_gram_lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        conversions = {}
        for sent in text:
            doc = nlp(' '.join(sent))
            for token in doc:
                if len(token.lemma_) <4 and str(token)!=str(token.lemma_):
                    conversions[token] = token.lemma_
    #                 print('%s : %s' (token, token.lemma_))
        return conversions

    def init_n_gram_mods(processed_docs):
        np.random.seed(2018)
        nltk.download('wordnet')
        nlp = spacy.load('en', disable=['parser', 'ner'])

        stemmer = SnowballStemmer('english')
        STOP_WORDS = list(gensim.parsing.preprocessing.STOPWORDS)
        STOP_WORDS.extend(['table', 'legend'])

        # - Trigram model : implement n-grams with Gensim Phrases
        #min_count = ignore all words and bigrams with total collected count lower than this value
        #threshold = represent a score threshold for forming the phrases(higher means fewer phrases)
        bigram = gensim.models.Phrases(processed_docs, min_count=1, threshold=1)
        trigram = gensim.models.Phrases(bigram[processed_docs], threshold=1)

        #sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        return bigram_mod, trigram_mod
