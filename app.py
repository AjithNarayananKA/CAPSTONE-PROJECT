# Step - 1 - Importing Libraries

import streamlit as st
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import PyPDF2
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Step - 2 - Set the model paths 

# Here we specifies the path to a pre-trained model ("lora-flan-t5-large-qg") by combining the current file's directory with
# the "models" folder. It also loads a sentence embedding model called all-MiniLM-L6-v2 using the SentenceTransformer library
# to convert sentences into numerical vectors, and initializes the spaCy NLP pipeline using the en_core_web_sm model for
# processing English text.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lora-flan-t5-large-qg")
SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')


# Here we defines a function called extract_text_from_pdf, which extracts text from a given PDF file. It uses the PyPDF2 library
# to read the PDF, then loops through each page and extracts the text from it, appending it to a string. Finally,
# it returns the combined text from all the pages in the PDF.

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Here we defines a function called extract_skills that identifies and extracts specific skills from a given text. It uses the spaCy
# library to process the text, converting it to lowercase for consistency. A predefined list of common skills 
# (like 'python', 'java', 'machine learning', etc.) is checked against the text. If any of these skills are found in the text,
# they are added to a set to avoid duplicates. Finally, the function returns the list of identified skills.

def extract_skills(text):
    """Extract skills from text using spaCy and predefined skill patterns"""
    # Add your skill extraction logic here based on your ipynb file
    # This is a placeholder implementation
    doc = nlp(text.lower())
    
    # Example skill patterns - replace with your comprehensive list
    common_skills = ['python', 'java', 'javascript', 'sql', 'machine learning', 
                     'deep learning', 'nlp', 'data analysis', 'aws']
    
    found_skills = set()
    for skill in common_skills:
        if skill in text.lower():
            found_skills.add(skill)
            
    return list(found_skills)


# Here we defines a function called generate_model_answer, which generates a detailed, concise, and non-repetitive answer to a given
# interview question using a language model. It creates a prompt with instructions for the answer, which is then tokenized
# and passed into the model. The model generates a response with parameters to control repetition, length, and sampling.
# After generation, the function processes the text to remove any repeated sentences by comparing normalized sentence structures.
# The final cleaned answer is returned, ensuring it ends with appropriate punctuation.

def generate_model_answer(question, model, tokenizer, device):
    """Generate a model answer for a given interview question with no repetition"""
    prompt = f"""Human: Provide a clear and detailed technical answer to this interview question: {question}
Rules for the answer:
- Provide complete technical details but avoid repetition
- Keep the explanation focused and concise
- Use examples or code snippets where relevant
- Maintain a clear structure in the explanation
Assistant:"""
    
    input_ids = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        do_sample=True,
        top_p=0.9,
        max_length=256,
        temperature=0.7,
        min_length=50,
        max_new_tokens=200,
        no_repeat_ngram_size=5,  # Increased to prevent phrase repetition
        repetition_penalty=1.5,   # Added repetition penalty
        length_penalty=0.8        # Slightly penalize longer sequences
    )
    
    answer = tokenizer.batch_decode(
        outputs.detach().cpu().numpy(), 
        skip_special_tokens=True
    )[0]
    
    # Post-process to remove repetitions
    def remove_repeated_sentences(text):
        # Split into sentences
        sentences = text.split('. ')
        # Remove duplicates while preserving order
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            # Normalize sentence for comparison
            normalized = sentence.lower().strip()
            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_sentences.append(sentence)
        return '. '.join(unique_sentences)
    
    # Clean up the answer
    cleaned_answer = remove_repeated_sentences(answer)
    # Ensure the answer ends with proper punctuation
    if cleaned_answer and not cleaned_answer.endswith(('.', '?', '!')):
        cleaned_answer += '.'
    
    return cleaned_answer.strip()


# Here we defines a function generate_interview_question that generates a single technical interview question based on a 
# given skill and difficulty level. It uses predefined templates to create prompts for beginner, intermediate, advanced,
# or expert-level questions. The prompt is processed by a language model, which generates the question while ensuring
# specific controls to avoid repetition. After generation, the function cleans the text by ensuring the question ends with
# a single question mark and removes any repeated phrases. The final question, along with the skill and difficulty level,
# is returned in a structured format.

def generate_interview_question(skill, difficulty, model, tokenizer, device):
    """Generate a specific single technical interview question based on skill and difficulty"""
    
    prompt_templates = {
        "Beginner": "Generate a single basic technical interview question about {skill} suitable for beginners. The question should focus on one specific fundamental concept.",
        "Intermediate": "Generate a single intermediate-level technical interview question about {skill}. Focus on one specific practical implementation scenario.",
        "Advanced": "Generate a single advanced technical interview question about {skill} that tests deep understanding of one specific complex concept.",
        "Expert": "Generate a single expert-level technical interview question about {skill} focusing on one specific aspect of system design or architecture."
    }
    
    prompt = prompt_templates[difficulty].format(skill=skill)
    
    full_prompt = f"""Human: {prompt}
Rules for the question:
- Generate only ONE question, not multiple
- End the question with a single question mark
- Focus on a single specific aspect or concept
- Make the question clear and concise
Assistant:"""
    
    input_ids = tokenizer(
        full_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        do_sample=True,
        top_p=0.9,
        max_length=256,
        num_return_sequences=1,
        temperature=0.7,
        min_length=20,
        max_new_tokens=100,
        no_repeat_ngram_size=5,    # Increased to prevent phrase repetition
        repetition_penalty=1.5,    # Added repetition penalty
        length_penalty=0.8         # Slightly penalize longer sequences
    )
    
    question = tokenizer.batch_decode(
        outputs.detach().cpu().numpy(), 
        skip_special_tokens=True
    )[0].strip()
    
    # Post-process to ensure single question
    def clean_question(text):
        # Split by question marks and take only the first question
        if '?' in text:
            text = text.split('?')[0] + '?'
        # Remove any repeated phrases
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            if i + 3 < len(words):  # Check for repeated 3-word phrases
                phrase1 = ' '.join(words[i:i+3])
                phrase2 = ' '.join(words[i+3:i+6])
                if phrase1.lower() == phrase2.lower():
                    i += 6  # Skip the repeated phrase
                    continue
            cleaned_words.append(words[i])
            i += 1
        return ' '.join(cleaned_words)
    
    cleaned_question = clean_question(question)
    
    return {
        "skill": skill,
        "question": cleaned_question,
        "difficulty": difficulty
    }


# Here we defines a function called calculate_similarity that measures how similar two answers are by using sentence embeddings.
# It encodes both answers into numerical vectors using a pre-trained sentence embedding model (SENTENCE_MODEL).
# The embeddings are reshaped and then compared using cosine similarity. The resulting similarity score is returned as a percentage.

def calculate_similarity(answer1, answer2):
    """Calculate similarity between two answers using sentence embeddings"""
    # Encode the answers
    embedding1 = SENTENCE_MODEL.encode([answer1])[0]
    embedding2 = SENTENCE_MODEL.encode([answer2])[0]
    
    # Reshape embeddings for cosine_similarity
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    return similarity * 100  # Convert to percentage


# Here we defines a function load_model that loads a pre-trained model and tokenizer from a local directory using caching for efficiency.
# It retrieves the model configuration from a specified path (MODEL_PATH), loads the model and tokenizer,
# and sets the device to either GPU (cuda) or CPU based on availability.
# The model is then wrapped using the PeftModel for efficient adaptation and placed on the appropriate device.
# Finally, the function returns the loaded model, tokenizer, and device.

@st.cache_resource
def load_model():
    """Load the model and tokenizer from local directory"""
    config = PeftConfig.from_pretrained(MODEL_PATH)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PeftModel.from_pretrained(
        model, 
        MODEL_PATH,
        device_map={"": 0} if device == "cuda" else None
    ).to(device)
    model.eval()
    
    return model, tokenizer, device


# Here we defines a function generate_interview_question that creates a single technical interview question based on
# a given skill and difficulty level. It uses predefined prompt templates for different difficulty levels
# (Beginner, Intermediate, Advanced, Expert) to generate a focused question.
# The prompt is processed by a language model, which generates the question while following rules to ensure
# it’s clear, concise, and only one question. The response is post-processed to keep only the first question
# if multiple questions are generated. The function returns the skill, the generated question, and its difficulty level.

def generate_interview_question(skill, difficulty, model, tokenizer, device):
    """Generate a specific single technical interview question based on skill and difficulty"""
    
    # Difficulty-specific prompt templates - modified to emphasize single question
    prompt_templates = {
        "Beginner": "Generate a single basic technical interview question about {skill} suitable for beginners. The question should focus on one specific fundamental concept.",
        "Intermediate": "Generate a single intermediate-level technical interview question about {skill}. Focus on one specific practical implementation scenario.",
        "Advanced": "Generate a single advanced technical interview question about {skill} that tests deep understanding of one specific complex concept.",
        "Expert": "Generate a single expert-level technical interview question about {skill} focusing on one specific aspect of system design or architecture."
    }
    
    # Get the appropriate prompt template
    prompt = prompt_templates[difficulty].format(skill=skill)
    
    # Add specific instruction to ensure single question format
    full_prompt = f"""Human: {prompt}
Rules for the question:
- Generate only ONE question, not multiple
- End the question with a single question mark
- Focus on a single specific aspect or concept
- Avoid compound questions with multiple parts
Assistant:"""
    
    input_ids = tokenizer(
        full_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        do_sample=True,
        top_p=0.9,
        max_length=256,
        num_return_sequences=1,
        temperature=0.7,
        min_length=20,
        max_new_tokens=100,  # Limit response length
        no_repeat_ngram_size=3
    )
    
    question = tokenizer.batch_decode(
        outputs.detach().cpu().numpy(), 
        skip_special_tokens=True
    )[0].strip()
    
    # Post-process to ensure single question
    # Split by question marks and take only the first question
    if '?' in question:
        question = question.split('?')[0] + '?'
    
    return {
        "skill": skill,
        "question": question,
        "difficulty": difficulty
    }


# Here we defines a main function for a web app using Streamlit, which serves as an AI-powered interview assistant. 
# It allows users to upload a resume, extract skills from it, and generate technical interview questions based on
# selected skills and difficulty levels. The app has two main sections: Resume Analysis
# (where users upload a PDF resume and extract skills) and Interview Practice
# (where users select a skill and difficulty level, then generate and answer interview questions).
# After answering, the app compares the user's answer with a model-generated answer and 
# calculates a similarity score. The chat history, including questions, answers, and similarity scores,
# is displayed for reference. Error handling ensures the app runs smoothly.

def main():
    st.title("AI - Based Interview Question Generator")
    
    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "extracted_skills" not in st.session_state:
        st.session_state.extracted_skills = []
    
    try:
        with st.spinner("Loading models... This might take a minute."):
            model, tokenizer, device = load_model()
        
        tab1, tab2 = st.tabs(["Resume Analysis", "Interview Practice"])
        
        with tab1:
            st.header("Resume Analysis")
            uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf", key="resume_uploader")
            
            if uploaded_file is not None:
                resume_text = extract_text_from_pdf(uploaded_file)
                skills = extract_skills(resume_text)
                st.session_state.extracted_skills = skills
                
                st.subheader("Extracted Skills")
                st.write(", ".join(skills))
        
        with tab2:
            st.header("Interview Practice")
            
            if not st.session_state.extracted_skills:
                st.write("Please upload a resume first to extract skills!")
            else:
                # Create columns for skill and difficulty selection
                col1, col2 = st.columns(2)
                
                with col1:
                    # Domain/Skill selection
                    selected_skill = st.selectbox(
                        "Select Domain/Skill:",
                        st.session_state.extracted_skills,
                        key="skill_selector"
                    )
                
                with col2:
                    # Difficulty selection
                    selected_difficulty = st.selectbox(
                        "Select Difficulty Level:",
                        ["Beginner", "Intermediate", "Advanced", "Expert"],
                        key="difficulty_selector"
                    )
                
                # Generate question button
                if st.button("Generate New Question", key="generate_question_btn"):
                    with st.spinner("Generating question..."):
                        question_data = generate_interview_question(
                            selected_skill,
                            selected_difficulty,
                            model,
                            tokenizer,
                            device
                        )
                        st.session_state.current_question = question_data
                
                # Display current question and answer section
                if st.session_state.current_question:
                    st.markdown("""---""")
                    st.markdown(f"""
                    **Current Question** ({st.session_state.current_question['difficulty']})  
                    **Domain:** {st.session_state.current_question['skill']}  
                    {st.session_state.current_question['question']}
                    """)
                    
                    # Answer input
                    user_answer = st.text_area("Your Answer:", height=100, key="answer_input")
                    
                    # Check answer button
                    if st.button("Check Answer", key="check_answer_btn"):
                        if user_answer:
                            with st.spinner("Generating model answer and calculating similarity..."):
                                model_answer = generate_model_answer(
                                    st.session_state.current_question['question'],
                                    model,
                                    tokenizer,
                                    device
                                )
                                
                                similarity_score = calculate_similarity(user_answer, model_answer)
                                
                                st.subheader("Results")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Your Answer:**")
                                    st.write(user_answer)
                                    
                                with col2:
                                    st.markdown("**Model Answer:**")
                                    st.write(model_answer)
                                
                                # Display similarity score with color coding
                                score_color = "red" if similarity_score < 50 else "orange" if similarity_score < 75 else "green"
                                st.markdown(f"**Similarity Score:** <span style='color:{score_color}'>{similarity_score:.2f}%</span>", unsafe_allow_html=True)
                                
                                # Add to chat history
                                st.session_state.chat_history.append(("Question", st.session_state.current_question['question']))
                                st.session_state.chat_history.append(("Difficulty", st.session_state.current_question['difficulty']))
                                st.session_state.chat_history.append(("Domain", st.session_state.current_question['skill']))
                                st.session_state.chat_history.append(("Your Answer", user_answer))
                                st.session_state.chat_history.append(("Model Answer", model_answer))
                                st.session_state.chat_history.append(("Similarity", f"{similarity_score:.2f}%"))
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("""---""")
            st.subheader("Interview History")
            for i in range(0, len(st.session_state.chat_history), 6):  # Group entries by 6
                st.markdown(f"""
                **Question:** {st.session_state.chat_history[i][1]}  
                **Difficulty:** {st.session_state.chat_history[i+1][1]}  
                **Domain:** {st.session_state.chat_history[i+2][1]}  
                **Your Answer:** {st.session_state.chat_history[i+3][1]}  
                **Model Answer:** {st.session_state.chat_history[i+4][1]}  
                **Similarity Score:** {st.session_state.chat_history[i+5][1]}
                """)
                st.markdown("---")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure you have all required models and dependencies installed.")



# Here we calls the main() function to execute the application. 

if __name__ == "__main__":
    main()        
