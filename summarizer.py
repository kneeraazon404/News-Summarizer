import os
import requests
import json
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Retrieve API keys from environment variables
news_api_key = os.getenv("NEWS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo-16k"

# Initialize the OpenAI client
client = OpenAI()


def get_news(topic):
    """
    Fetches news articles based on a topic using the NewsAPI.

    Parameters:
    - topic (str): The topic to fetch news articles about.

    Returns:
    - List of dictionaries containing news article details. Empty list if an error occurs.
    """
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    )

    try:
        response = requests.get(url)
        if response.status_code == 200:
            news_json = response.json()
            articles = news_json["articles"]
            results = []
            for article in articles:
                items = {
                    "title": article["title"],
                    "author": article["author"],
                    "url": article["url"],
                    "source": article["source"]["name"],
                    "content": article["content"],
                    "description": article["description"],
                }
                results.append(items)
            return results
        else:
            logging.error("Failed to fetch news")
            return []

    except Exception as e:
        logging.error("Failed to fetch news: %s", e)
        return []


def summarize_news(articles):
    """
    Summarizes the content of news articles.

    Parameters:
    - articles (list): A list of dictionaries each containing details of a news article.

    Returns:
    - List of dictionaries with the article title and its summary.
    """
    summaries = []
    for article in articles:
        if article["content"]:
            summary = client.summarize(article["content"], model)
            summaries.append({"title": article["title"], "summary": summary})
    return summaries


class AssistantManager:
    """
    Manages interactions with the OpenAI assistant, including creating assistants, threads, and processing messages.
    """

    thread_id = None
    assistant_id = None

    def __init__(self, model=model):
        self.client = client
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None

        # Retrieve existing assistant and thread if IDs are already created
        if AssistantManager.assistant_id:
            self.assistant = self.client.beta.assistant.retrieve(
                assistant_id=AssistantManager.assistant_id
            )
        if AssistantManager.thread_id:
            self.thread = self.client.beta.assistant.thread.retrieve(
                thread_id=AssistantManager.thread_id
            )

    def create_assistant(self, name, instructions, tools):
        """
        Creates a new assistant if one does not already exist.

        Parameters:
        - name (str): The name of the assistant.
        - instructions (str): Instructions for the assistant.
        - tools (list): A list of tools the assistant can use.
        """
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name, instructions=instructions, tools=tools, models=self.model
            )
            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            logging.info("Assistant ID: %s", self.assistant.id)

    def create_thread(self):
        """
        Creates a new thread if one does not already exist.
        """
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id
            self.thread = thread_obj
            logging.info("Thread ID: %s", self.thread.id)

    def add_message_to_thread(self, role, content):
        """
        Adds a message to the thread.

        Parameters:
        - role (str): The role of the message sender.
        - content (str): The content of the message.
        """
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id, role=role, content=content
            )

    def run_assistant(self, instructions):
        """
        Initiates the assistant to process given instructions.

        Parameters:
        - instructions (str): The instructions for the assistant to process.
        """
        if self.thread and self.assistant:
            self.run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions,
            )

    def process_messages(self):
        """
        Processes messages from the thread and compiles a summary.
        """
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)

            summary = []
            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)

            self.summary = "\n".join(summary)
            logging.info(
                "SUMMARY ::::::=============> , ROLE : %s :==============> %s",
                role.capitalize(),
                response,
            )

            for msg in messages:
                role = msg.role
                content = msg.content[0].text.value
                logging.info("SUMMARY ================>: %s", content)

    def wait_for_completion(self):
        """
        Waits for the assistant's processing to complete, handling any required actions.
        """
        if self.thread and self.run:
            while True:
                time.sleep(5)  # Polling interval
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id, run_id=self.run.id
                )

                logging.info("RUN STATUS: %s", run_status.model_dump_json(indent=4))

                if run_status.status == "completed":
                    self.process_messages()
                    break
                elif run_status.status == "requires_action":
                    logging.info("FUNCTION CALLING NOW...")
                    self.call_required_functions(
                        required_actions=run_status.required_action.submit_tool_outputs.model_dump()
                    )

    def call_required_functions(self, required_actions):
        """
        Calls required functions based on the assistant's needs.

        Parameters:
        - required_actions (dict): Actions required by the assistant.
        """
        if not self.run:
            return
        tool_outputs = []
        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(topic=arguments["topic"])
                logging.info("STUFF: %s", output)

                final_str = "".join(
                    str(item) for item in output
                )  # Fixed concatenation of dict items

                tool_outputs.append({"tool_call_id": action["id"], "output": final_str})
            else:
                raise ValueError("Unknown function: %s", func_name)

            logging.info("Submitting outputs back to the Assistant...")
            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id,
                run_id=self.run.id,
                tool_outputs=tool_outputs,
            )

    def get_summary(self):
        """
        Retrieves the summary of processed messages.

        Returns:
        - str: The compiled summary.
        """
        return self.summary

    def run_steps(self):
        """
        Retrieves and logs the steps run by the assistant.
        """
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id, run_id=self.run.id
        )
        logging.info("RUN STEPS: %s", run_steps)


def main():
    """
    Main function to fetch news on a topic and print the first article.
    """
    news = get_news("Crypto")
    if news:
        logging.info(news[0])
    else:
        logging.info("No news articles found.")


if __name__ == "__main__":
    main()
