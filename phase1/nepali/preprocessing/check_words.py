import json
try:
    with open('nepalese_scraper/nepali_news_data.txt.state.json', 'r') as f:
        data = json.load(f)
        print(data.get('total_words', 0))
except Exception as e:
    print(e)
