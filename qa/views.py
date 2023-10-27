#Old Code
"""from django.shortcuts import render
from textblob import TextBlob
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from transformers import ( TrainingArguments, AdamW,T5ForConditionalGeneration,T5Tokenizer,get_linear_schedule_with_warmup)
import nltk
import re
import PyPDF2
from nltk.tokenize import sent_tokenize
def preprocess_text(text):

    # Text preprocessing
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9(){}[],\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # Leading and trailing white space removed
def sentences(text):
    sent_list = sent_tokenize(text)  # Renamed the variable to avoid conflict
    return sent_list
    # Sentence tokenization
  # Tokenize the text into sentences
def answer_question(question, context):
    # Initialize the DistilBERT tokenizer and model
    from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

    # Load the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad",truncation=True, padding='max_length', max_length=512, return_tensors="pt", model_max_length=512)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad",return_dict=False)

    # Tokenize question and context
    tokenized_input = tokenizer(question, context, truncation=True, padding='max_length', max_length=512, return_tensors="pt", model_max_length=512)
    # Get the input_ids and attention_mask tensors
    input_ids = tokenized_input["input_ids"]
    attention_mask = tokenized_input["attention_mask"]
    # Perform QA inference
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer_tokens = input_ids[0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)
    return answer
def extract_text_from_txt(text_file):
    try:
        text = ""
        for chunk in text_file.chunks():
            text += chunk.decode('utf-8')
        return text
    except Exception as e:
        return ""  # Return an empty string in case of an error


# Modify qa_view to use the text extraction function
def qa_view(request):
    if request.method == 'POST':
        context = request.POST.get('context')
        txt_file = request.FILES.get('txt_file')  # Change pdf_file to txt_file
        if txt_file:  # Change the file type check
            txt_text = extract_text_from_txt(txt_file)  # Use the text extraction function
            if txt_text:
                context = txt_text
        question1 = request.POST.get('question1', '')
        answer1 = answer_question(question1, context)
        question2 = request.POST.get('question2', '')
        answer2 = answer_question(question2, context)
        # Repeat the above for question3, question4, question5
        def perform_sentiment_analysis(text):
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            return polarity, subjectivity
        def perform_pos_tagging(text):
            blob = TextBlob(text)
            pos_tags = blob.tags
            return pos_tags

        return render(
            request,
            'qa/qa_template.html',
            {
                'context': context,
                'question1': question1,
                'answer1': answer1,
                'question2': question2,
                'answer2': answer2,
                # Pass the remaining questions and answers as well
            }
        )

    return render(request, 'qa/qa_template.html', {'context': ''})
"""

#Updated code
from django.shortcuts import render
from textblob import TextBlob
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import HttpResponse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from django.views.decorators.csrf import csrf_protect
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from transformers import ( TrainingArguments, AdamW,T5ForConditionalGeneration,T5Tokenizer,get_linear_schedule_with_warmup)
import nltk
import re
import PyPDF2
from nltk.tokenize import sent_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(tokens)
    return text
def sentences(text):
     sent_list = sent_tokenize(text)  # Renamed the variable to avoid conflict
     return sent_list
     # Sentence tokenization
   # Tokenize the text into sentences
def answer_question(question, context):

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad",truncation=True, padding='max_length', max_length=512, return_tensors="pt", model_max_length=512)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad",return_dict=False)
    tokenized_input = tokenizer(question, context, truncation=True, padding='max_length', max_length=512, return_tensors="pt", model_max_length=512)
    input_ids = tokenized_input["input_ids"]
    attention_mask = tokenized_input["attention_mask"]
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer_tokens = input_ids[0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)
    return answer
def extract_text_from_txt(text_file):
    try:
        text = ""
        for chunk in text_file.chunks():
            text += chunk.decode('utf-8')
            return text
    except Exception as e:
        return ""
def qa_view(request):
    if request.method == 'POST':
        context = request.POST.get('context')
        txt_file = request.FILES.get('txt_file')  # Change pdf_file to txt_fil
        if txt_file:
            txt_text = extract_text_from_txt(txt_file)  # Use the text extraction function
            if txt_text:

                context = txt_text
        question1 = request.POST.get('question1', '')
        answer1 = answer_question(question1, context)
        question2 = request.POST.get('question2', '')
        answer2 = answer_question(question2, context)
        question3 = request.POST.get('question3', '')
        answer3 = answer_question(question3, context)
        question4 = request.POST.get('question4', '')
        answer4 = answer_question(question4, context)
        question5 = request.POST.get('question5', '')
        answer5 = answer_question(question5, context)
        def perform_sentiment_analysis(text):
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            return polarity, subjectivity

        def perform_pos_tagging(text):
             blob = TextBlob(text)
             pos_tags = blob.tags
             return pos_tags
        return render(request,
             'qa/qa_template.html',
             {
                 'context': context,
                 'question1': question1,
                 'answer1': answer1,
                 'question2': question2,
                 'answer2': answer2,
                 'question3': question3,
                 'answer3': answer3,
                 'question4': question4,
                 'answer4': answer4

                 })

    return render(request, 'qa/qa_template.html', {'context': ''})

