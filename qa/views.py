from django.shortcuts import render
from django.http import HttpResponse
#from .forms import PDFUploadForm
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import nltk
from nltk.tokenize import sent_tokenize
from django.shortcuts import render, redirect
#from .models import ProcessedText  # Import the model you defined
#import fitz  # PyMuPDF

def upload_text(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        text = request.POST.get('text')
        processed_text = preprocess_text(text)  # Preprocess the text
        ProcessedText.objects.create(title=title, text=processed_text)  # Store the processed text in the database
        return redirect('text_list')  # Replace 'text_list' with the URL name for listing text passages

    return render(request, 'upload_text.html')

def preprocess_text(text):
    # Text preprocessing
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9(){}[],\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # Leading and trailing white space removed

    # Sentence tokenization
    sentences = sent_tokenize(text)
  # Tokenize the text into sentences
def answer_question(question, context):
    # Initialize the DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

    # Tokenize question and context
    tokenized_input = tokenizer(question, context, truncation=True, padding='max_length', max_length=512, return_tensors="pt")

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
def qa_view(request):
    if request.method == 'POST':
        context = request.POST.get('context')

        # Check if a PDF file has been uploaded
        pdf_file = request.FILES.get('pdf_file')
        if pdf_file:
            # Extract text from the uploaded PDF
            pdf_text = extract_text_from_pdf(pdf_file)
            # If text extraction is successful, use it as the context
            if pdf_text:
                context = pdf_text

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

"""def extract_text_from_pdf(pdf_file):
    try:
        # Initialize a PyMuPDF document
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

        # Extract text from each page
        pdf_text = ""
        for page in pdf_document:
            pdf_text += page.get_text()

        return pdf_text
    except Exception as e:
        # Log the exception for debugging
        print(f"PDF Extraction Error: {e}")
        return "Error: Unable to extract PDF text"""

## for uploading pdf files ##
"""def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('pdf_list')  # Replace 'pdf_list' with the URL name for listing PDFs

    else:
        form = PDFUploadForm()

    return render(request, 'upload_pdf.html', {'form': form})"""
