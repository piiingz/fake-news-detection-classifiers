IDUN = True

RANDOM_SEED = 2021
TEST_SIZE = 0.15
VAL_SIZE = 0.10
TUNING_SIZE = 0.3

# Dataset sizes
LABELED_VALIDATION_ABSOLUTE_SIZE = 345
LABELED_TRAIN_SIZE = 1380
WEAK_NUMBER_OF_ARTICLES_PER_CLASS = 2760


# TF-IDF params
TITLE_MAX_FEATURES = 1000
CONTENT_MAX_FEATURES = 5000

# LR params
MAX_ITER = 4000
SOLVER = 'liblinear'
PENALTY = 'l1'
CV = 5

# LR Tuned params
WEAK_BEST_C = 5
SUPERVISED_BEST_C = 1

# BERT config
BERT_N_WORDS = 512

# Numerical cols
NUMERICAL_COLS = ['id',
                  'content_exclamation_count',
                  'title_exclamation_count',
                  'content_word_count',
                  'title_word_count',
                  'content_word_count_with_punctuation',
                  'title_word_count_with_punctuation',
                  'content_sentence_count',
                  'title_sentence_count',
                  'content_capital_word_count',
                  'title_capital_word_count',
                  'content_stop_word_count',
                  'title_stop_word_count',
                  'content_stop_word_ratio',
                  'title_stop_word_ratio',
                  'content_words_per_sentence',
                  'content_quote_marks_count',
                  'content_ttr_score',
                  'title_ttr_score',
                  'title_nouns_count',
                  'title_proper_nouns_count',
                  'content_avg_word_length',
                  'title_avg_word_length',
                  'content_avg_word_length_no_stop_words',
                  'title_avg_word_length_no_stop_words',
                  'content_url_count',
                  'content_verb_ratio',
                  'content_past_tense_verb_ratio',
                  'content_past_tense_verb_ratio_of_all_verbs',
                  'content_adjective_ratio',
                  'content_adverb_ratio',
                  'title_verb_ratio',
                  'title_past_tense_verb_ratio',
                  'title_past_tense_verb_ratio_of_all_verbs',
                  'title_adjective_ratio',
                  'title_adverb_ratio',
                  'content_capital_word_ratio',
                  'title_capital_word_ratio',
                  'content_personal_pronouns_count',
                  'content_personal_pronouns_ratio',
                  'content_quote_marks_ratio',
                  'title_nouns_ratio',
                  'title_proper_nouns_ratio',
                  'content_exclamation_ratio',
                  'title_exclamation_ratio',
                  'content_sentiment_word_sub',
                  'content_sentiment_word_pos',
                  'content_sentiment_word_neg',
                  'title_sentiment_word_sub',
                  'title_sentiment_word_pos',
                  'title_sentiment_word_neg',
                  'content_sentiment_sentence_sub',
                  'content_sentiment_sentence_pos',
                  'content_sentiment_sentence_neg',
                  'title_sentiment_sentence_sub',
                  'title_sentiment_sentence_pos',
                  'title_sentiment_sentence_neg',
                  'content_sentiment_text_sub',
                  'content_sentiment_text_pos',
                  'content_sentiment_text_neg',
                  'title_sentiment_text_sub',
                  'title_sentiment_text_pos',
                  'title_sentiment_text_neg',
                  'title_swn_pos_score',
                  'title_swn_neg_score',
                  'title_swn_obj_score',
                  'content_swn_pos_score',
                  'content_swn_neg_score',
                  'content_swn_obj_score'
                  ]


LEMMATIZED_COLS = ['id', 'content_lemmatized_lowercase_no_stopwords', 'title_lemmatized_lowercase_no_stopwords']
RAW_TEXT_COLS = ['id', 'content', 'title']

LABELED_LEMMATIZED_COLS = ['id', 'content_lemmatized_lowercase_no_stopwords', 'title_lemmatized_lowercase_no_stopwords', 'label']
LABELED_RAW_TEXT_COLS = ['id', 'content', 'title', 'label']

BERT_SAVE_COLS = ['text', 'label']
