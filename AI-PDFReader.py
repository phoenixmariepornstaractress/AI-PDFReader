import os
import smtplib
import argparse
import logging
from tqdm import tqdm
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pyttsx3
import PyPDF2
import spacy
import pytesseract
from PIL import Image
from langdetect import detect
from keybert import KeyBERT
from datasets import load_dataset
from transformers import (
    pipeline, MarianMTModel, MarianTokenizer, BertTokenizer, BertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nlpaug.augmenter.word import SynonymAug, ContextualWordEmbsAug
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path, start_page=0, end_page=None):
    try:
        pdf_reader = PyPDF2.PdfFileReader(open(pdf_path, 'rb'))
    except (FileNotFoundError, PyPDF2.utils.PdfReadError):
        print("The specified PDF file was not found or could not be read.")
        return None

    if end_page is None or end_page > pdf_reader.numPages:
        end_page = pdf_reader.numPages

    full_text = ""
    for page_num in tqdm(range(start_page, end_page), desc="Extracting text"):
        page = pdf_reader.getPage(page_num)
        text = page.extractText().strip().replace('\n', ' ')
        full_text += text + " "

    return full_text

def text_to_speech(text, mp3_file, rate=150, volume=1.0, voice_id=0):
    speaker = pyttsx3.init()
    speaker.setProperty('rate', rate)
    speaker.setProperty('volume', volume)

    voices = speaker.getProperty('voices')
    if 0 <= voice_id < len(voices):
        speaker.setProperty('voice', voices[voice_id].id)
    else:
        print("Invalid voice ID. Using default voice.")

    speaker.save_to_file(text, mp3_file)
    speaker.runAndWait()
    speaker.stop()

def list_voices():
    speaker = pyttsx3.init()
    voices = speaker.getProperty('voices')
    for index, voice in enumerate(voices):
        print(f"Voice ID: {index}")
        print(f"Name: {voice.name}")
        print(f"Language: {voice.languages}")
        print(f"Gender: {voice.gender}")
        print(f"Age: {voice.age}")
        print("-" * 20)
    speaker.stop()

def extract_metadata(pdf_path):
    try:
        pdf_reader = PyPDF2.PdfFileReader(open(pdf_path, 'rb'))
        return pdf_reader.getDocumentInfo()
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None

def send_email_notification(to_email, subject, body):
    from_email = "your-email@example.com"
    password = "your-email-password"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

def summarize_text(text, max_length=130):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extract_key_phrases(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return list(set(chunk.text for chunk in doc.noun_chunks))

def translate_text(text, target_language='es'):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    translation = translator(text, max_length=512)
    return translation[0]['translation_text']

def named_entity_recognition(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [(entity.text, entity.label_) for entity in doc.ents]

def sentiment_analysis(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment = sentiment_analyzer(text)
    return sentiment[0]

def classify_document(text):
    classifier = pipeline("zero-shot-classification")
    candidate_labels = ["business", "technology", "science", "health", "entertainment"]
    return classifier(text, candidate_labels)

def extract_text_with_ocr(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text

def extract_keywords(text):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return keywords

def extract_topics(text, num_topics=5, num_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = [
        f"Topic {topic_idx + 1}: {' '.join(words[i] for i in topic.argsort()[:-num_words - 1:-1])}"
        for topic_idx, topic in enumerate(lda.components_)
    ]
    return topics

def detect_and_translate(text, target_language='es'):
    detected_language = detect(text)
    if detected_language != target_language:
        model_name = f'Helsinki-NLP/opus-mt-{detected_language}-{target_language}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    return text

def train_document_classifier(train_dataset, val_dataset, output_dir='doc_classification_model'):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_summarization_model(train_dataset, val_dataset, output_dir='summarization_model'):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def augment_text(text, method='synonym'):
    aug = SynonymAug(aug_src='wordnet') if method == 'synonym' else ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    return aug.augment(text)

def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF to Speech Converter with AI capabilities")
    parser.add_argument('pdf_path', type=str, help="Path to the PDF file")
    parser.add_argument('--start_page', type=int, default=0, help="Start page (0-indexed)")
    parser.add_argument('--end_page', type=int, help="End page")
    parser.add_argument('--output_dir', type=str, default='.', help="Output directory for the MP3 file")
    parser.add_argument('--mp3_file', type=str, help="Name of the output MP3 file")
    parser.add_argument('--voice_id', type=int, default=0, help="Voice ID")
    parser.add_argument('--rate', type=int, default=150, help="Speech rate")
    parser.add_argument('--volume', type=float, default=1.0, help="Volume (0.0 to 1.0)")
    parser.add_argument('--email', type=str, help="Email address to send notification")
    parser.add_argument('--summarize', action='store_true', help="Summarize the extracted text")
    parser.add_argument('--key_phrases', action='store_true', help="Extract key phrases from the text")
    parser.add_argument('--translate', action='store_true', help="Translate the text to Spanish")
    parser.add_argument('--ner', action='store_true', help="Perform Named Entity Recognition (NER) on the text")
    parser.add_argument('--sentiment', action='store_true', help="Perform sentiment analysis on the text")
    parser.add_argument('--classify', action='store_true', help="Classify the document into categories")
    parser.add_argument('--ocr', action='store_true', help="Extract text using OCR")
    parser.add_argument('--keywords', action='store_true', help="Extract keywords from the text")
    parser.add_argument('--topics', action='store_true', help="Extract topics from the text")
    parser.add_argument('--augment', choices=['synonym', 'contextual'], help="Augment the text")

    args = parser.parse_args()
    pdf_path = args.pdf_path

    if args.ocr:
        text = extract_text_with_ocr(pdf_path)
    else:
        text = extract_text_from_pdf(pdf_path, args.start_page, args.end_page)

    if not text:
        print("No text extracted from the PDF.")
        return

    if args.summarize:
        text = summarize_text(text)

    if args.key_phrases:
        key_phrases = extract_key_phrases(text)
        print("Key Phrases:", key_phrases)

    if args.translate:
        text = translate_text(text)

    if args.ner:
        entities = named_entity_recognition(text)
        print("Named Entities:", entities)

    if args.sentiment:
        sentiment = sentiment_analysis(text)
        print("Sentiment Analysis:", sentiment)

    if args.classify:
        classification = classify_document(text)
        print("Document Classification:", classification)

    if args.keywords:
        keywords = extract_keywords(text)
        print("Keywords:", keywords)

    if args.topics:
        topics = extract_topics(text)
        print("Topics:", topics)

    if args.augment:
        text = augment_text(text, method=args.augment)
        print("Augmented Text:", text)

    mp3_file = args.mp3_file or os.path.join(args.output_dir, "output.mp3")
    text_to_speech(text, mp3_file, rate=args.rate, volume=args.volume, voice_id=args.voice_id)

    if args.email:
        send_email_notification(args.email, "PDF to Speech Conversion Complete", f"Your MP3 file is ready: {mp3_file}")

if __name__ == "__main__":
    main()
