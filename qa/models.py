import nltk
from django.db import models
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

"""class QAData(models.Model):
    context = models.TextField()
    question1 = models.CharField(max_length=255, default="Default Answer for Question 1")
    answer1 = models.TextField()
    question2 = models.CharField(max_length=255, default="Default Answer for Question 2")
    answer2 = models.TextField()
    question3 = models.CharField(max_length=255, default="Default Answer for Question 3")
    answer3 = models.TextField(default="Default Answer for Question 3")
    question4 = models.CharField(max_length=255, default="Default Answer for Question 4")
    answer4 = models.TextField(default="Default Answer for Question 4")
    question5 = models.CharField(max_length=255, default="Default Answer for Question 5")
    answer5 = models.TextField(default="Default Answer for Question 5")
    sentiment_polarity = models.FloatField(null=True, blank=True)
    sentiment_subjectivity = models.FloatField(null=True, blank=True)
"""

class QAData(models.Model):
    context = models.TextField()
    question1 = models.CharField(max_length=255)
    answer1 = models.TextField()
    question2 = models.CharField(max_length=255)
    answer2 = models.TextField()
    question3 = models.CharField(max_length=255)
    answer3 = models.TextField()
    question4 = models.CharField(max_length=255)
    answer4 = models.TextField()
    question5 = models.CharField(max_length=255)
    answer5 = models.TextField()
    sentiment_polarity = models.FloatField(null=True, blank=True)
    sentiment_subjectivity = models.FloatField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.sentiment_polarity or not self.sentiment_subjectivity:
            blob = TextBlob(self.context)
            self.sentiment_polarity = blob.sentiment.polarity
            self.sentiment_subjectivity = blob.sentiment.subjectivity
        super().save(*args, **kwargs)

    @staticmethod
    def answer_question(question, context):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad", truncation=True, padding='max_length', max_length=512, return_tensors="pt", model_max_length=512)
        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad", return_dict=False)
        tokenized_input = tokenizer(question, context, truncation=True, padding='max_length', max_length=512, return_tensors="pt", model_max_length=512)
        input_ids = tokenized_input["input_ids"]
        attention_mask = tokenized_input["attention_mask"]
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        answer_tokens = input_ids[0][start_index: end_index + 1]
        answer = tokenizer.decode(answer_tokens)
        return answer

    @staticmethod
    def preprocess_text(text):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        text = ' '.join(tokens)
        return text



