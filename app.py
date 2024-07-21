from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool, tool
import streamlit as st
import os
from dotenv import load_dotenv
import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from IPython.display import Markdown

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.4.1/bin/tesseract'

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)
search_tool = SerperDevTool(api_key=SERPER_API_KEY)

# create a root window
root = tk.Tk()
root.withdraw()

# open the file dialog box
image_path = filedialog.askopenfilename()


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    return denoised_image


processedImage = preprocess_image(image_path)


def extract_text_from_image(processedImage):
    text = pytesseract.image_to_string(processedImage, lang='eng')
    return text


recognized_text = extract_text_from_image(processedImage)


def create_agent(role, goal, backstory):
    return Agent(
        llm=llm,
        role=role,
        goal=goal,
        backstory=backstory,
        allow_delegation=False,
        verbose=True,
    )


newsAnalyst = create_agent(
    role="News Analyst",
    goal="Analyze news and articles and get important insights and retrieve relevant information from {news}",
    backstory="You are a news analyst and your job is to analyze {news} from different newspapers, create a"
              " summary and understand what is happening in your country. Your job is to forward these important news "
              "to government ministers so that they can take important decision",
)

newsEditor = create_agent(
    role="News Editor",
    goal="You edit the {news} and articles before publishing",
    backstory="You are a news editor and you job is to edit the {news}. You delete or edit any content that you "
              "think is not good for public, provides balanced viewpoints, and avoids major controversial topics ",
)


def create_task(description, expected_output, agent):
    return Task(description=description, expected_output=expected_output, agent=agent)


analyze = create_task(
    description=(
      "1. Analyze the news, important points and purpose from the {news}.\n"
      "2. Identify the target audience and psychological effects of {news} on them.\n"
      "3. Create a summary and also another summary of your analysis of {news}.\n"
    ),
    expected_output= "Summary of news and also summary of your analysis of the news, main keywords and "
                     "author name",
    agent=newsAnalyst,
)

edit = create_task(
    description="Proofread the given {news} post for grammatical errors, edit the content "
                " that is not good for public mental health and brand voice alignment.",
    expected_output="A well written news, free of errors, ready for publication, with each "
                    "section having 2-3 paragraphs.",
    agent=newsEditor,
)


crew = Crew(agents=[newsAnalyst, newsEditor], tasks=[analyze, edit], verbose=2)

st.title("AI News Articles Analyzer")

if st.button("Start Workflow"):
    with st.spinner("Running the content creation workflow..."):
        result = crew.kickoff(inputs={"news": recognized_text})
    st.write(result)
    st.success("Workflow completed!")
