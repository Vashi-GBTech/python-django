# qa/views.py
from django.shortcuts import render
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

def answer_question(question, context):
    # Initialize the DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

    encoding = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = input_ids[0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    return answer

def qa_view(request):
    if request.method == 'POST':
        context = request.POST.get('context')

        question1 = request.POST.get('question1', '')
        answer1 = answer_question(question1, context)

        question2 = request.POST.get('question2', '')
        answer2 = answer_question(question2, context)

        # Repeat the above for question3, question4, question5

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
    return render(request, 'qa/qa_template.html')

