from src.models.newsclassifier import NewsClassifier
from src.utils.get_path import get_path

ROOT = get_path(2)
save_model_path = ROOT/"src"/'saved_model'

news_classifier = NewsClassifier(save_model_path/"trained_lg_model.pkl",save_model_path/'mapping.pkl',save_model_path/'vectorizer.pkl' )

print('classifier running, enter `exit` to quit')
while True:
    news_headline = input('Enter a news headline to classify:')
    if news_headline == 'exit':
        break
    classification = news_classifier.classify_review(news_headline)

    print("Classification Report:",classification)
