import openai
import os
from dotenv import load_dotenv
from openai import OpenAI
import warnings
import tiktoken

load_dotenv()

tokenizer = tiktoken.get_encoding('cl100k_base')


openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
global history, pro

warnings.filterwarnings("ignore")

prompt = """
        You are an expert Islamic Chatbot tasked with answering any question about Islam finance.
        Generate a comprehensive and informative answer of 100 words or less for the \
        given question based solely on the provided information (content). You must \
        only use information from the provided information. Use an unbiased and \
        Islamic Finance Scholar tone. Combine information provided together into a coherent answer. Do not \
        repeat text.
        <context>
            {text} 
        <context/>
    """

suffix = """
{history}
query: {input}
answer: 

"""

example_template = """
query: {query}
answer: {answer}
"""

exit_text = "Sorry, The context is not present in the information."

examples = [
    {
        "query": "How did they evaluated the incontext learning approach?",
        "answer": """We evaluate the proposed approach on several natural language understanding and generation benchmarks, where the retrieval-based prompt selection approach consistently out performs the random baseline. Moreover, it
        is observed that the sentence encoders fine tuned on task-related datasets yield even more helpful retrieval results""",
    },
    {
        "query": "On Which type of task incontext learning task achieved success?",
        "answer": """ Notably, significant gains are observed on tasks such as table-to text generation (41.9% on the ToTTo dataset) and open-domain question answering (45.5% on the NQ dataset). We hope our investigation could help understand the behaviors of GPT-3 and large-scale pre-trained LMs in general and enhance their few-shot capabilities.""",
    },
]


def prompting(text, previous_query_st):
    prompt_hadith = f"""You are an expert Islamic  Chatbot tasked with answering any question about Islamic finance. 
Generate a concise or comprehensive or informative answer depending on the question, based solely on the provided information (context). 
You must only use information from the provided context. Use an unbiased and Islamic Scholar tone. 
Combine information provided together into a coherent answer without repeating text.

Instructions:
    - If the answer is not present in the context, respond with "Not Present."
    - In case there is contradicting information in any of the documents provided, ensure your answer acknowledges the different perspectives. Clearly explain the different opinions and state that various approaches may have different views.



Here is the context you should use to answer the questions:

<context>
{text}
</context>
    """

    prompt = f"""You are an expert Islamic Chatbot tasked with answering any question about Islamic finance. Generate a concise or comprehensive or informative answer depending on the question, based solely on the provided information (content). You must only use information from the provided information. Use an unbiased and Islamic Scholar tone. Combine information provided together into a coherent answer without repeating text. If there is a reference present in the information provided, then provide a reference (Verse, Chapter number, or Chapter name or Quran Verse) in the answer and make sure to enclose it in square brackets ([]).

In case there is contradicting information in any of the documents provided, ensure your answer acknowledges the different perspectives. Clearly explain the different opinions and state that various approaches may have different views.


There may be a case in which its a follow up question based on the previous question so here is a follow up question and Answer for your reference in case its required while answering the question
Previous Conversation:
{previous_query_st}
if the query says "Are you Sure" reply with yes I am sure with a quick brief about why it is a correct answer based on the previous conversation.

If the answer is not present in the context, respond with "Not Present."

Here is the context you should use to answer the questions:

<context>
    {text} 
<context/>
        """
    return prompt, prompt_hadith


completion_kwargs = {
    "model": "gpt-4o-mini",
    "max_tokens": 1200,
    "temperature": 0.1,
}

stream_completion_kwargs = {**completion_kwargs, "stream": True}

stream_completion_kwargs_with_usage = {
    **completion_kwargs,
    "stream": True,
    "stream_options": {"include_usage": True},
}
