import os

from dotenv import load_dotenv
from openai import OpenAI
import requests
import json

load_dotenv()


news_api_key = os.environ.get("NEWS_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-3.5-turbo-16k"

client = OpenAI()


def get_news(topic):
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    )

    try:
        response = requests.get(url)
        if response.status_code == 200:
            final_news = []
            news = json.dumps(response.json(), indent=4)
            news_json = json.loads(news)
            articles = news_json["articles"]
            for article in articles:
                title = article["title"]
                source_name = article["source"]["name"]
                author = article["author"]
                description = article["description"]
                content = article["content"]
                url = article["url"]
                results = f"""
                Title: {title}
                author: {author}
                description: {description}
                source name: {source_name}
                url: {url}
                content: {content}
                
                """

                final_news.append(results)
            return final_news

        else:
            print("Failed to fetch news")
            return []

    except Exception as e:
        print("Failed to fetch news")
        print(e)
        return []


def summarize_news(articles):
    summaries = []
    for article in articles:
        title = article["title"]
        content = article["content"]
        if content:
            summary = client.summarize(content, model)
            summaries.append({"title": title, "summary": summary})
    return summaries


if __name__ == "__main__":
    news = get_news("Crypto")
    print(news[0])
    # summaries = summarize_news(articles)
    # for summary in summaries:
    #     print(summary["title"])
    #     print(summary["summary"])
    #     print("\n")
