import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pypdf import PdfReader
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# Load existing QA pairs
def load_qa_pairs(json_path):
    with open(json_path, 'r') as f:
        qa_data = json.load(f)
    
    # Convert to list of training examples
    training_examples = []
    for question, answer in qa_data.items():
        training_examples.append({
            "question": question,
            "answer": answer
        })
    
    print(f"Loaded {len(training_examples)} QA pairs from JSON")
    return training_examples

#  Split the extracted text into chunks 
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

text_chunks = create_chunks("")

#  Generate Q&A pairs 
def generate_qa_pairs_from_chunks(chunks):
    qa_pairs = []
    
    for chunk in chunks:
        sentences = chunk.page_content.split(".")  # Split text into sentences
        for sentence in sentences:
            if len(sentence) > 30:  # Ensure sentences are long enough for Q&A
                question = f"What is discussed in the following sentence? {sentence.strip()}"
                answer = sentence.strip()
                qa_pairs.append({"question": question, "answer": answer})
    
    return qa_pairs

qa_pairs = generate_qa_pairs_from_chunks(text_chunks)

# Save the Q&A pairs to a JSON file
def save_qa_pairs_to_json(qa_pairs, filename="qa_pairs.json"):
    with open(filename, "w") as f:
        json.dump(qa_pairs, f, indent=4)

# Extract text from PDF book
def extract_book_content(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        extracted_text = []
        for i in range(total_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                extracted_text.append(text)
                
        print(f"Successfully extracted {len(extracted_text)} pages")
        # Initialize vector database
        embeddingModel = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorDbPath = "vectorstore/db_faiss"
        os.makedirs(vectorDbPath, exist_ok=True)
        vectorDb = FAISS.load_local(vectorDbPath, embeddingModel, allow_dangerous_deserialization=True)

        print("\nCreating vector store for retrieval...")
        # Create QA chain
        qaChain = RetrievalQA.from_chain_type(
        llm=train_model(),
        chain_type="stuff",
        retriever=vectorDb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(
        template="""
                Answer the following question based on the context provided.

                Context: {context}
                Question: {question}

                Answer:
                """,
        input_variables=["context", "question"]
        )})
        return qaChain
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

# Convert book content to training examples
def generate_training_pairs_from_book(book_text):
    training_pairs = []
    
    # Process each page
    for page_text in book_text:
        # Split into paragraphs (text chunks)
        paragraphs = [p for p in page_text.split('\n\n') if len(p.strip()) > 50]
        
        for paragraph in paragraphs:
            # Create context-based questions
            words = paragraph.split()
            if len(words) < 10:
                continue
                
            # Generate questions based on paragraph content
            # 1. General information question
            training_pairs.append({
                "question": f"What happens in this part of Harry Potter: '{paragraph[:50]}...'?",
                "answer": paragraph
            })
            
            # 2. Character-focused questions if character names are detected
            character_names = ["Harry", "Ron", "Hermione", "Dumbledore", "Snape", "Voldemort"]
            for name in character_names:
                if name in paragraph:
                    training_pairs.append({
                        "question": f"What does {name} do in this scene?",
                        "answer": paragraph
                    })
    
    print(f"Generated {len(training_pairs)} training examples from book content")
    return training_pairs

def prepare_dataset_for_training(training_examples):
    # Convert to Dataset object
    dataset = Dataset.from_pandas(pd.DataFrame(training_examples))
    
    # Split into train/validation sets (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1)
    
    return dataset

def tokenize_dataset(dataset, tokenizer, max_input_length=512, max_target_length=128):
    # Prefix for T5 question-answering
    prefix = "answer: "
    
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["question"]]
        targets = examples["answer"]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=max_input_length, 
            padding="max_length", 
            truncation=True
        )
        
        # Setup the targets
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["question", "answer"]
    )
    
    return tokenized_dataset

def train_model(tokenized_dataset, model_name="t5-small", output_dir="./modelCheckpoints"):
    # Load base model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer


def main():
    print("🧙 Harry Potter Knowledge Base - Training Pipeline")
    print("=================================================")
    
    # 1. Load QA pairs from JSON
    qa_pairs = load_qa_pairs("data/harry_potter_qa.json")
    
    # 2. Extract book content
    book_path = "data/harry_potter_book.pdf"
    if os.path.exists(book_path):
        book_content = extract_book_content(book_path)
        book_examples = generate_training_pairs_from_book(book_content)
        
        # Combine both data sources
        all_training_examples = qa_pairs + book_examples
        print(f"Combined dataset contains {len(all_training_examples)} examples")
    else:
        print(f"Book not found at {book_path}. Using only QA pairs.")
        all_training_examples = qa_pairs
    
    # 3. Prepare dataset
    dataset = prepare_dataset_for_training(all_training_examples)
    
    # 4. Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # 5. Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # 6. Train model
    model, tokenizer = train_model(tokenized_dataset)
    
    print("saving model artifact in to './modelCheckpoint'...")
    print("Saved Successfully!")
    print("\n✅ Training complete! You can now use your model with the Streamlit app.")

if __name__ == "__main__":
    main()
