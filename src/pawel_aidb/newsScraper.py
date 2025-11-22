import requests


URLS = [
    "https://thepeoplesvoice.tv/",
    "https://www.justfactsdaily.com/50-examples-of-fake-news-in-2024"
]

for url in URLS:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open("articles.txt","wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)