from src.config import tokenizer
from src.services import logservice
from src.prompt import prompts_on_source
from src.services.chromaservice import ChromaService
from src.config import exit_text
import requests
import numpy as np
from src.services import openaiservice
from openai import APIConnectionError, RateLimitError, APIStatusError, APIError
class FinanceILM():
    def __init__(self) -> None:
        pass

    def tiktoken_len(self,text:str) ->int:
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    


    async def get_context(self, question: str, source: str):
        """
        Retrieves context from the Chroma pipeline.

        Args:
            question (str): The question to be sent.
            source (str): The source for the context.

        Returns:
            tuple: A tuple containing the text and the extracted link, or (None, None) if an error occurs.
        """
        logservice.logging.info("Starting get_context function to retrieve context from the Chroma pipeline.")
        chromasvc = ChromaService()
        try:
            print("question",question)
            text_l, link_extracted, score_l = await chromasvc.get_context_info_optimized(question, source)
            
        except requests.exceptions.HTTPError as http_err:
            logservice.logging.error("HTTP error occurred: %s", http_err)
            return None, None, None
        except requests.exceptions.RequestException as req_err:
            logservice.logging.error("Request exception occurred: %s", req_err)
            return None, None, None
        except Exception as err:
            logservice.logging.error("An unexpected error occurred: %s", err)
            return None, None, None
        else:
            try:
                text = '\n\n'.join(text_l)
                score = float(np.mean(score_l)) if score_l else 0
                logservice.logging.info("Context received successfully.")
                return text, link_extracted, score
            except ValueError as json_err:
                logservice.logging.error("Error parsing JSON response: %s", json_err)
                return None, None, None


    def format_last_queries(self, data):
        """
        Extracts and formats the last four question-answer pairs from a dictionary.

        Args:
            data (dict): A dictionary containing a "queries" key with a list of question-answer pairs.

        Returns:
            str: A formatted string containing the last four (or fewer) question-answer pairs.
                Each pair is formatted as:

                Question: <Question>
                Answer: <Answer>
        """
        try:
            # Extract the list of queries
            queries = data.get("queries", [])
            
            # Ensure 'queries' is a list
            if not isinstance(queries, list):
                raise TypeError("'queries' should be a list.")
            
            # Determine how many queries to extract (max 4)
            num_queries = min(len(queries), 4)
            
            # Get the last `num_queries` pairs
            last_queries = queries[-num_queries:]
            
            # Format them into the required form
            formatted_output = ""
            for query in last_queries:
                try:
                    question = query['Question']
                    answer = query['Answer']
                    formatted_output += f"Question: {question}\nAnswer: {answer}\n"
                except KeyError as e:
                    logservice.logging.error("Missing expected key in query: %s", e)
                    continue  # Skip this query and proceed to the next one
            
            logservice.logging.info("Successfully formatted last queries.")
            return formatted_output.strip()
        
        except TypeError as type_err:
            logservice.logging.error("Type error occurred: %s", type_err)
        except Exception as ex:
            logservice.logging.error("An unexpected error occurred: %s", ex)
        
        return ""
        
                

    def rephrase_query(self, data_input, question, flag: int = 0) -> str:
        """
        Rephrases a follow-up question into a standalone question for FinanceILM.

        Args:
            data_input (dict): Conversation history data.
            question (str): The recent user question to be rephrased.
            flag (int, optional): Reserved/unused flag. Defaults to 0.

        Returns:
            str: The rephrased standalone question, or an empty string if an error occurs.
        """
        try:
            previous_query_str = self.format_last_queries(data_input) or ""
            logservice.logging.debug("Formatted previous conversations successfully.")

            # Domain-focused, strict output, no hallucinations
            prompt_rephrase = (
                "You are a FinanceILM assistant that REPHRASES follow-up questions into a single, "
                "clear, standalone question specifically about ISLAMIC FINANCE topics (e.g., Shariah-compliant "
                "products, contracts, screens, regulatory/compliance rules, AAOIFI/IFSB guidance, local "
                "jurisdictional regulations, sukuk, takaful, murabaha, ijara, mudaraba, musharaka, etc.).\n\n"
                "Requirements:\n"
                "- Do NOT add new facts. Do NOT infer details that aren't present.\n"
                "- Resolve pronouns like 'it/that/this' using the chat history when possible.\n"
                "- Keep any mentioned entity names, instruments, jurisdictions, dates, currencies, and thresholds.\n"
                "- Prefer neutral, compliance-aware phrasing. If the follow-up is ambiguous, keep it conservative but standalone.\n"
                "- Output ONLY the rephrased question text. No labels, no quotes, no explanations.\n\n"
                "Examples (illustrative):\n"
                "User context: 'Compare murabaha vs ijara for asset financing in KSA.'\n"
                "Follow-up: 'Which one fits leasing better?'\n"
                "Rephrased: 'Which contract fits leasing better in Saudi Arabia, murabaha or ijara?'\n"
                "----\n"
                "User context: 'AAOIFI screening ratios for equities.'\n"
                "Follow-up: 'What about thresholds for cash and receivables?'\n"
                "Rephrased: 'What are the AAOIFI equity screening thresholds for cash and receivables?'\n"
                "----\n"
                "User context: 'Takaful vs conventional insurance regulatory differences in Malaysia.'\n"
                "Follow-up: 'And disclosures?'\n"
                "Rephrased: 'What disclosure requirements differentiate takaful from conventional insurance in Malaysia?'\n"
            )

            response = openaiservice.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_rephrase},
                    {"role": "user", "content": f"Chat history:\n{previous_query_str}"},
                    {"role": "user", "content": f"Follow-up to rephrase:\n{question}\n\nReturn ONLY the standalone question."},
                ],
                max_tokens=80,
                temperature=0.0,
            )

            # Defensive parsing
            content = ""
            if hasattr(response, "choices") and response.choices:
                msg = getattr(response.choices[0], "message", None)
                if msg and hasattr(msg, "content") and msg.content:
                    content = msg.content.strip()

            if not content:
                # If the model returned nothing, keep behavior predictable
                logservice.logging.warning("Empty content received for rephrased query; returning empty string.")
                return ""

            # Strip any accidental prefixes the model might add
            cleaned = (
                content.replace("Rephrased:", "")
                    .replace("REPHRASED:", "")
                    .replace("Rephrased Query:", "")
                    .replace("Standalone Query:", "")
                    .strip(" \n:‘’\"")
                    .strip()
            )

            logservice.logging.info("Received rephrased query successfully.")
            print("REPHRASED Query:", cleaned)  # Optional: keep for local debugging

            return cleaned

        except Exception as e:
            logservice.logging.error("An error occurred while rephrasing the query: %s", e)
            return ""




    def Answer_Generator(self, text, score, data_input, recent_query, new_query, source):
        """
        Generates an answer using an AI model based on the provided inputs.

        Args:
            text (str): The context or text input for the model.
            data_input (dict): The conversation history data.
            recent_query (str): The most recent query from the user.
            new_query (str): The rephrased query to be used.
            source (str): The source information for generating the prompt.

        Returns:
            tuple: A tuple containing the final response from the AI model and the raw response object.
        """
        try:
            previous_query_str = self.format_last_queries(data_input)
            logservice.logging.debug("Formatted previous conversations successfully.")

            prompt = prompts_on_source(source, text, score)
            logservice.logging.debug("Generated prompt successfully.")

            response = openaiservice.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "system", "content": f"Recent Query: {recent_query}"},
                    {"role": "user", "content": f"Rephrased Query: {new_query}"}
                ],
                max_tokens=1200,
                temperature=0.1,
            )
            final_response = response.choices[0].message.content.strip()
            logservice.logging.info("Received response from AI model successfully.")

        except APIConnectionError as e:
            logservice.logging.error("The server could not be reached: %s", e)
            final_response = "Sorry, there was an issue connecting to the server. Please refresh and try again."
            response = None
        except RateLimitError as e:
            logservice.logging.error("Rate limit exceeded: %s", e)
            final_response = "Sorry, we are receiving too many requests. Please try again later."
            response = None
        except APIError as e:
            logservice.logging.error("An API error occurred: %s", e)
            final_response = "Sorry, there was an issue processing your request. Please refresh and try again."
            response = None
        except Exception as e:
            logservice.logging.error("An unexpected error occurred: %s", e)
            final_response = "Sorry, an unexpected error occurred. Please try again later."
            response = None

        return final_response, response




    def Answer_Generator_without_memory(self, text, score, question, source):
        """
        Generates an answer using an AI model based on the provided inputs, without using conversation memory.

        Args:
            text (str): The context or text input for the model.
            question (str): The question to be answered.
            source (str): The source information for generating the prompt.

        Returns:
            tuple: A tuple containing the final response from the AI model and the raw response object.
        """
        try:
            prompt = prompts_on_source(source, text, score)
            logservice.logging.debug("Generated prompt successfully.")

            response = openaiservice.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Query: {question}"}
                ],
                max_tokens=1200,
                temperature=0.1,
            )
            final_response = response.choices[0].message.content.strip()
            logservice.logging.info("Received response from AI model successfully.")

        except APIConnectionError as e:
            logservice.logging.error("The server could not be reached: %s", e)
            final_response = "Sorry, there was an issue connecting to the server. Please refresh and try again."
            response = None
        except RateLimitError as e:
            logservice.logging.error("Rate limit exceeded: %s", e)
            final_response = "Sorry, we are receiving too many requests. Please try again later."
            response = None
        except APIError as e:
            logservice.logging.error("An API error occurred: %s", e)
            final_response = "Sorry, there was an issue processing your request. Please refresh and try again."
            response = None
        except Exception as e:
            logservice.logging.error("An unexpected error occurred: %s", e)
            final_response = "Sorry, an unexpected error occurred. Please try again later."
            response = None

        return final_response, response



    #### Streaming Code###############################################

    def Answer_Generator_stream(self,text, score, data_input, recent_query, new_query, source):
        """
    Generates an answer using an AI model based on the provided inputs, streaming the response.

    Args:
        text (str): The context or text input for the model.
        data_input (dict): The conversation history data.
        recent_query (str): The most recent query from the user.
        new_query (str): The rephrased query to be used.
        source (str): The source information for generating the prompt.

    Yields:
        str: Chunks of the AI model's response as they are received.
    """
        previous_query_str=self.format_last_queries(data_input)
        logservice.logging.debug("Formatted previous conversations successfully.")
        prompt= prompts_on_source(source, text, score)
        logservice.logging.debug("Generated prompt successfully.")
        try:
            response = openaiservice.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "system", "content": f"Recent Query{recent_query}"},
                    {"role": "user", "content":f"Rephrased Query{new_query}"}

                ],
                max_tokens=1200,
                temperature=0.1,
                stream = True
            )
            
            for chunk in response:
                if chunk is not None:
                    chunk = chunk.choices[0].delta.content
                    if chunk is None:
                        continue
                    chunk = str(chunk)
                    # print(chunk)
                    yield chunk
        except APIConnectionError as e:
            logservice.logging.error("API connection error: %s", e)
            yield "Sorry, there was an issue connecting to the server. Please refresh and try again."
        except RateLimitError as e:
            logservice.logging.error("Rate limit exceeded: %s", e)
            yield "Sorry, we are receiving too many requests. Please try again later."
        except APIError as e:
            logservice.logging.error("An API error occurred: %s", e)
            yield "Sorry, there was an issue processing your request. Please refresh and try again."
        except Exception as e:
            logservice.logging.error("An unexpected error occurred: %s", e)
            yield "Sorry, an unexpected error occurred. Please try again later."

    def Answer_Generator_without_memory_stream(self,text, score, question, source):
        """
    Generates an answer using an AI model based on the provided inputs, without using conversation memory, streaming the response.

    Args:
        text (str): The context or text input for the model.
        question (str): The question to be answered.
        source (str): The source information for generating the prompt.

    Yields:
        str: Chunks of the AI model's response as they are received.
    """
        prompt= prompts_on_source(source, text, score)
        logservice.logging.debug("Generated prompt successfully.")
        try:
            response = openaiservice.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "system", "content": f"Query{question}"}

                ],
                max_tokens=1200,
                temperature=0.1,
                stream = True
            )
            for chunk in response:
                if chunk is not None:
                    chunk = chunk.choices[0].delta.content
                    if chunk is None:
                        continue
                    chunk = str(chunk)
                    yield chunk
        except APIConnectionError as e:
            logservice.logging.error("API connection error: %s", e)
            yield "Sorry, there was an issue connecting to the server. Please refresh and try again."
        except RateLimitError as e:
            logservice.logging.error("Rate limit exceeded: %s", e)
            yield "Sorry, we are receiving too many requests. Please try again later."
        except APIError as e:
            logservice.logging.error("An API error occurred: %s", e)
            yield "Sorry, there was an issue processing your request. Please refresh and try again."
        except Exception as e:
            logservice.logging.error("An unexpected error occurred: %s", e)
            yield "Sorry, an unexpected error occurred. Please try again later."