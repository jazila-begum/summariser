# summarize_pdf.py
import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from utils.pdf_utils import extract_text_from_pdf
from utils.summarize_utils import clean_text, chunk_text_by_tokens, summarize_chunks, recursive_summarize, create_bullet_points

def summarize_pdf_with_study_aids(pdf_path, depth="medium"):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    print(f"Extracting text from PDF: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    print("Loading model and tokenizer...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    print("Chunking text by tokens...")
    chunks = list(chunk_text_by_tokens(cleaned_text, tokenizer, max_tokens=1024))

    # Depth configuration
    depth_config = {
        "short": {"min_length": 30, "max_length": 150},
        "medium": {"min_length": 80, "max_length": 400},
        "detailed": {"min_length": 150, "max_length": 700},
    }
    config = depth_config.get(depth.lower(), depth_config["medium"])
    print(f"Summarizing {len(chunks)} chunks with depth '{depth}'...")

    summaries = summarize_chunks(chunks, summarizer, tokenizer, **config, num_beams=6)

    print("Creating bullet points for chunk summaries...")
    bullets_only = [create_bullet_points(s) for s in summaries]

    print("Generating full recursive summary...")
    full_summary = recursive_summarize([" ".join(summaries)], summarizer, tokenizer, **config)

    return {
        "full_summary": full_summary,
        "bullets_by_chunk": bullets_only,
    }

if __name__ == "__main__":
    pdf_file = os.path.join("data", "yourfile.pdf")  # Update with your file name

    # Prompt user for summary depth
    print("Choose summary depth:")
    print("1. Short\n2. Medium\n3. Detailed")
    depth_input = input("Enter option (1/2/3): ").strip()

    depth_map = {
        "1": "short",
        "2": "medium",
        "3": "detailed"
    }

    depth_choice = depth_map.get(depth_input, "medium")  # default to medium if invalid
    print(f"\nYou selected: {depth_choice} depth\n")

    results = summarize_pdf_with_study_aids(pdf_file, depth=depth_choice)

    print("\n=== OVERALL SUMMARY ===\n", results["full_summary"])
    print("\n=== BULLET POINTS BY CHUNK ===\n")
    for bullets in results["bullets_by_chunk"]:
        print(bullets, "\n")
