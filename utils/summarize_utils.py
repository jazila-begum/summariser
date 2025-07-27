# utils/summarize_utils.py

import re
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def clean_text(text):
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text_by_tokens(text, tokenizer, max_tokens=1024):
    tokens = tokenizer.tokenize(text)
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i: i + max_tokens]
        yield tokenizer.convert_tokens_to_string(chunk_tokens)

def summarize_chunks(chunks, summarizer, tokenizer, min_length=30, max_length=300, num_beams=4):
    model = summarizer.model
    device = model.device
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)

        summary_ids = model.generate(
            **inputs,
            min_length=min_length,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries

def recursive_summarize(text_chunks, summarizer, tokenizer, min_length=30, max_length=300):
    summaries = summarize_chunks(text_chunks, summarizer, tokenizer, min_length, max_length)
    combined = " ".join(summaries)
    if len(combined.split()) > 750:
        return recursive_summarize(chunk_text_by_tokens(combined, tokenizer), summarizer, tokenizer, min_length, max_length)
    return combined

def create_bullet_points(summary_text):
    sentences = re.split(r"(?<=[.!?]) +", summary_text)
    bullets = ["- " + s.strip() for s in sentences if s.strip()]
    return "\n".join(bullets)
