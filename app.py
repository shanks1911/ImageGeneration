from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
from image import generate_image

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ('user', "Give me a prompt to be passed into an image generation model which represents the quote: {question}. The length of the prompt should be less than 50 words.")
    ]
)

# Streamlit framework
st.title("Generation of Prompts for Image generation model using Ollama Llama3.2")
input_text = st.text_input("Enter the Quote.")

# Ollama LLM
llm = Ollama(model="llama3.2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    generated_prompt = chain.invoke({'question': input_text})  # Correct assignment

    # Check if generated_prompt is valid
    if generated_prompt and isinstance(generated_prompt, str):
        st.write("Generated Prompt:", generated_prompt)

        # Generate the image
        image_path = generate_image(generated_prompt)
        
        # Display the generated image
        st.image(image_path, caption="Generated Image")
    else:
        st.error("The generated prompt is empty or not valid.")
