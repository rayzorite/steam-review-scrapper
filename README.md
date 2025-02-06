## Steam Review Scraper

### Overview
A Python-based tool to scrape Steam game reviews, analyze sentiment, and extract insights using Natural Language Processing (NLP). The results are saved in CSV and JSON formats with visualizations.

### Features
- Scrape Steam reviews for a given game App ID.
- Two scraping modes: "balanced" (equal positive & negative reviews) and "random".
- Data cleaning & NLP processing (removal of stopwords, formatting, sentiment analysis).
- Frequent positive & negative words extraction using Opinion Lexicon.
- Data visualization (bar charts, word clouds, sentiment distribution plots).
- CSV & JSON output storage.

### Installation
Ensure the following dependencies are installed:
- pandas
- nltk
- wordcloud
- plotly
- selenium
- questionary
- rich

> Download using pip: `pip install pandas nltk wordcloud plotly selenium questionary rich`

### Usage
- run `python scrapper.py`
- Enter Game App ID.
- Choose scraping mode:
    - **balanced:** Equal positive & negative reviews.
    - **random:** Collects reviews randomly.
- Enter the number of reviews (default: 20).
- Selenium automates review collection.
- Data processing includes:
  - Extracting review text, recommendation status, playtime, and date.
  - Cleaning text, removing stopwords.
  - Performing sentiment analysis.
  - Identifying frequent positive & negative words.
- Data saved in result/ directory:
    - reviews.csv
    - reviews.json
- Visualizations generated:
  - Word clouds.
  - Bar charts (recommendations, most used words).
  - Box plots (sentiment distribution).

### Key Functions
`get_current_scroll_position(browser)` Returns vertical scroll offset.

`scroll_to_bottom(browser, progress_task=None)` Scrolls browser to load more reviews.

`scrape_review_data(card)` Extracts review details.

`remove_stopwords_from_text(text, stopword_set)` Cleans text.

`plot_wordcloud(series, output_filename='wordcloud')` Generates a word cloud.


### Scraping Modes
- Balanced: Equal positive & negative reviews.
- Random: Collects reviews randomly.

### Data Processing
- Converts date formats.
- Tokenizes and cleans text.
- Uses SentimentIntensityAnalyzer for polarity scores.
- Correlates sentiment with recommendations.

### Output Files
- reviews.csv: Structured review data.
- reviews.json: JSON format.
- most-used.png: Word cloud image.
- Interactive charts with plotly.express. (Opens in web browser)

### Error Handling
- Handles invalid App IDs.
- Manages Selenium exceptions.
- Closes browser on failure.

### Notes
- Requires your current Browser's WebDriver. (just use firefox its pain to explain)
- Scrapes English-language reviews only. (you can change it in the script)

### Future Improvements
- Headless Selenium mode.
- Advanced NLP for sentiment classification.
- GUI interface.
- dditional data sources (Metacritic, Reddit).

---

### 📌 Contributions are welcome! Feel free to fork & enhance.

---

Licensed under CC0, just please don't sell it :)

