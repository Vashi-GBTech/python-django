# Tesitng Model

from django.shortcuts import render
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

def answer_question(question, context):
    # Initialize the DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

    # Define the sliding window parameters
    window_size = 512  # Adjust this based on your needs
    overlap = 128  # Adjust this for overlapping windows

    answers = []

    for i in range(0, len(context), window_size - overlap):
        window = context[i:i + window_size]

        # Tokenize the question and window
        encoding = tokenizer.encode_plus(question, window, add_special_tokens=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)

        # Find the answer in the window
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        answer_tokens = input_ids[0][start_index: end_index + 1]
        answer = tokenizer.decode(answer_tokens)

        answers.append(answer)

    # Consolidate answers and remove duplicates
    unique_answers = list(set(answers))
    consolidated_answer = " ".join(unique_answers)

    return consolidated_answer

def qa_view(request):
    if request.method == 'POST':
        context = request.POST.get('context')

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

        # Repeat the above for question6, question7, question8, and so on...

        return render(
            request,
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
                'answer4': answer4,
                'question5': question5,
                'answer5': answer5,

                # Pass the remaining questions and answers as well
            }
        )
    return render(request, 'qa/qa_template.html')
