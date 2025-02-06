import time
import os
import json
from collections import Counter

import pandas as pd
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import opinion_lexicon
from wordcloud import WordCloud

import plotly.express as px
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.options import Options
from selenium import webdriver

# For interactive prompts using arrow keys
import questionary

# Import Rich for enhanced terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------
# User Input for App ID, Scraping Mode, and Review Count
# ---------------------------
app_id_input = questionary.text("Enter Game's App ID:").ask()
try:
    GAME_ID = int(app_id_input)
except ValueError:
    console.print("[bold red]Invalid App ID. Please enter a numeric App ID.[/]")
    exit(1)

# Using a dropdown (arrow key navigation) for the scraping mode
scrape_mode = questionary.select(
    "Choose scraping mode:",
    choices=["balanced", "random"]
).ask()

reviews_count_input = questionary.text("Enter the number of reviews to scrape (Default 20 Reviews):").ask()
try:
    total_reviews_to_scrape = int(reviews_count_input)
except ValueError:
    console.print("[bold yellow]Invalid number entered. Defaulting to 20 reviews.[/]")
    total_reviews_to_scrape = 20

# ---------------------------
# Configuration and Setup
# ---------------------------
URL_TEMPLATE = "https://steamcommunity.com/app/{}/reviews/?p=1&browsefilter=mostrecent&filterLanguage=english"
url = URL_TEMPLATE.format(GAME_ID)
console.print(Panel(f"[bold cyan]Scraping URL:[/] {url}", title="Steam Review Scraper"))

# Configure headless mode
options = Options()
options.add_argument("--headless")  # Enable headless mode

# Initialize Selenium WebDriver with headless option
console.print("[bold green]Initializing headless browser...[/]")
browser = webdriver.Firefox(options=options)
browser.get(url)

# Download necessary NLTK resources
console.print("[bold yellow]Downloading NLTK resources...[/]")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('opinion_lexicon', quiet=True)

# ---------------------------
# Helper Functions
# ---------------------------
def get_current_scroll_position(browser):
    """Return the current vertical scroll offset."""
    return browser.execute_script("return window.pageYOffset;")

def scroll_to_bottom(browser, progress_task=None):
    """Scroll the browser window to the bottom and update progress."""
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)
    if progress_task:
        progress_task.advance(1)  # Increment progress

def get_steam_id(card):
    """Extract and return the unique Steam ID from a review card."""
    try:
        profile_url = card.find_element(By.XPATH, './/div[@class="apphub_friend_block"]/a').get_attribute('href')
        steam_id = profile_url.split('/')[-1]
    except Exception:
        steam_id = None
    return steam_id

def scrape_review_data(card):
    """
    Extract review information from a review card and return:
    review text, recommendation text, review length, play hours, and date posted.
    """
    date_posted_element = card.find_element(By.XPATH, './/div[@class="apphub_CardTextContent"]/div[@class="date_posted"]')
    date_posted = date_posted_element.text.strip()

    try:
        compensation_text = card.find_element(By.CLASS_NAME, "received_compensation").text.strip()
    except NoSuchElementException:
        compensation_text = ""

    card_text_content = card.find_element(By.CLASS_NAME, "apphub_CardTextContent").text.strip()
    # Remove date and compensation text from the review content
    for exclusion in [date_posted, compensation_text]:
        card_text_content = card_text_content.replace(exclusion, "")
    review_text = card_text_content.replace("\n", " ").replace("\t", " ").strip()

    # Remove "EARLY ACCESS REVIEW" text from the review
    review_text = review_text.replace("EARLY ACCESS REVIEW", "").strip()

    review_length = len(review_text.replace(" ", ""))

    # This field holds the recommendation. It is assumed to be either "Recommended" or "Not Recommended".
    thumb_text = card.find_element(By.XPATH, './/div[@class="reviewInfo"]/div[2]').text.strip()
    play_hours = card.find_element(By.XPATH, './/div[@class="reviewInfo"]/div[3]').text.strip()

    return review_text, thumb_text, review_length, play_hours, date_posted

def remove_stopwords_from_text(text, stopword_set):
    """Remove English stopwords from a given text and return the cleaned text."""
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopword_set]
    return ' '.join(filtered_words)

def plot_wordcloud(series, output_filename='wordcloud'):
    """Generate and save a word cloud image from a pandas Series."""
    text = ' '.join(series.astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(f"{output_filename}.png")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# ---------------------------
# Create Output Folder
# ---------------------------
output_folder = "result"
os.makedirs(output_folder, exist_ok=True)
console.print(Panel(f"[bold green]Output folder created (if not exists):[/] {output_folder}", title="Folder Setup"))

# ---------------------------
# Scraping Reviews
# ---------------------------
if scrape_mode == 'balanced':
    positive_reviews = []
    negative_reviews = []
    seen_steam_ids = set()

    # In balanced mode, the target is the number per category.
    target_per_category = total_reviews_to_scrape
    max_scroll_attempts = 4
    scroll_attempt = 0
    scroll_count = 0

    console.print(Panel(f"[bold yellow]Started scraping for reviews (Balanced Mode - {target_per_category} per category)...[/]"))

    try:
        last_scroll_position = get_current_scroll_position(browser)
        while (len(positive_reviews) < target_per_category or len(negative_reviews) < target_per_category) and scroll_attempt < max_scroll_attempts:
            prev_total = len(positive_reviews) + len(negative_reviews)

            try:
                show_more_button = browser.find_element(By.CLASS_NAME, "btn_green_white_innerfade")
                show_more_button.click()
                console.print("[bold magenta]Clicked 'Show More' button...[/]")
                time.sleep(3)
            except NoSuchElementException:
                scroll_to_bottom(browser)
                scroll_count += 1  # Increment scroll counter
                console.print(f"[dim]({scroll_count}) Scrolled to bottom, Loading more...[/]")
                time.sleep(3)

            review_cards = browser.find_elements(By.CLASS_NAME, 'apphub_Card')

            for card in review_cards[-50:]:
                steam_id = get_steam_id(card)
                if steam_id and steam_id in seen_steam_ids:
                    continue

                review_data = scrape_review_data(card)
                recommendation = review_data[1].strip()
                if recommendation == "Recommended" and len(positive_reviews) < target_per_category:
                    positive_reviews.append(review_data)
                    seen_steam_ids.add(steam_id)
                elif recommendation == "Not Recommended" and len(negative_reviews) < target_per_category:
                    negative_reviews.append(review_data)
                    seen_steam_ids.add(steam_id)

            current_total = len(positive_reviews) + len(negative_reviews)
            if current_total == prev_total:
                scroll_attempt += 1
                console.print(f"[bold red]Failed to find more reviews. Scroll attempt {scroll_attempt}/{max_scroll_attempts}.[/]")
            else:
                scroll_attempt = 0

            last_scroll_position = get_current_scroll_position(browser)
            time.sleep(2)

        total_scraped = len(positive_reviews) + len(negative_reviews)
        console.print(Panel(f"[bold green]Total reviews scraped:[/] {total_scraped} (Positive: {len(positive_reviews)}, Negative: {len(negative_reviews)})", title="Scraping Results"))

    except Exception as e:
        console.print(Panel(f"[bold red]Error during scraping: {e}[/]", title="Error"))
    finally:
        browser.quit()
        console.print("[bold yellow]Browser closed.[/]")

    # Combine positive and negative reviews for further processing
    reviews = positive_reviews + negative_reviews

else:  # Random Mode
    all_reviews = []
    seen_steam_ids = set()
    max_scroll_attempts = 4
    scroll_attempt = 0
    scroll_count = 0

    console.print(Panel(f"[bold yellow]Started scraping for reviews (Random Mode - target: {total_reviews_to_scrape} reviews)...[/]"))

    try:
        last_scroll_position = get_current_scroll_position(browser)
        while len(all_reviews) < total_reviews_to_scrape and scroll_attempt < max_scroll_attempts:
            prev_total = len(all_reviews)

            try:
                show_more_button = browser.find_element(By.CLASS_NAME, "btn_green_white_innerfade")
                show_more_button.click()
                console.print("[bold magenta]Clicked 'Show More' button...[/]")
                time.sleep(3)
            except NoSuchElementException:
                scroll_to_bottom(browser)
                scroll_count += 1
                console.print(f"[dim]({scroll_count}) Scrolled to bottom, Loading more...[/]")
                time.sleep(3)

            review_cards = browser.find_elements(By.CLASS_NAME, 'apphub_Card')

            for card in review_cards[-50:]:
                steam_id = get_steam_id(card)
                if steam_id and steam_id in seen_steam_ids:
                    continue
                review_data = scrape_review_data(card)
                all_reviews.append(review_data)
                seen_steam_ids.add(steam_id)
                if len(all_reviews) >= total_reviews_to_scrape:
                    break  # Break if we have reached our target

            if len(all_reviews) == prev_total:
                scroll_attempt += 1
                console.print(f"[bold red]Failed to find more reviews. Scroll attempt {scroll_attempt}/{max_scroll_attempts}.[/]")
            else:
                scroll_attempt = 0

            last_scroll_position = get_current_scroll_position(browser)
            time.sleep(2)

        total_scraped = len(all_reviews)
        console.print(Panel(f"[bold green]Total reviews scraped:[/] {total_scraped}", title="Scraping Results"))

    except Exception as e:
        console.print(Panel(f"[bold red]Error during scraping: {e}[/]", title="Error"))
    finally:
        browser.quit()
        console.print("[bold yellow]Browser closed.[/]")

    reviews = all_reviews

# ---------------------------
# Data Processing and Cleaning
# ---------------------------
console.print(Panel("[bold blue]Processing and cleaning data...[/]"))

# Define column names: ReviewText, Review, ReviewLength, PlayHours, DatePosted
columns = ['ReviewText', 'Review', 'ReviewLength', 'PlayHours', 'DatePosted']
df = pd.DataFrame(reviews, columns=columns)

# Clean the PlayHours column by removing unnecessary text and stripping spaces
df['PlayHours'] = df['PlayHours'].str.replace("hrs on record", "", regex=False).str.strip()

# Standardize the DatePosted column
month_mapping = {
    'January': '01', 'February': '02', 'March': '03', 'April': '04',
    'May': '05', 'June': '06', 'July': '07', 'August': '08',
    'September': '09', 'October': '10', 'November': '11', 'December': '12'
}

date_extracted = df['DatePosted'].str.extract(r'(\d+)\s+(\w+)', expand=True)
df['Day'] = date_extracted[0]
df['Month'] = date_extracted[1].map(month_mapping)
df['DatePosted'] = pd.to_datetime(df['Day'] + '/' + df['Month'] + '/2024', format='%d/%m/%Y')
df['DatePosted'] = df['DatePosted'].dt.strftime('%d-%m-%Y')
df.drop(['Day', 'Month'], axis=1, inplace=True)

df['ReviewText'] = df['ReviewText'].str.strip()
df['Review'] = df['Review'].str.strip()

# ---------------------------
# Save Data to CSV and JSON
# ---------------------------
csv_filename = os.path.join(output_folder, 'reviews.csv')
df.to_csv(csv_filename, encoding='utf-8', sep=';', index=False)
console.print(Panel(f"[bold green]Cleaned CSV data saved to:[/] {csv_filename}", title="Data Saved"))

json_filename = os.path.join(output_folder, 'reviews.json')
data_as_records = df.to_dict(orient='records')
with open(json_filename, 'w', encoding='utf-8') as json_file:
    json.dump(data_as_records, json_file, ensure_ascii=False, indent=4)
console.print(Panel(f"[bold green]Cleaned JSON data saved to:[/] {json_filename}", title="Data Saved"))

# ---------------------------
# Text Processing and Analysis
# ---------------------------
console.print(Panel("[bold blue]Starting text processing and analysis...[/]"))

stopword_set = set(nltk_stopwords.words('english'))
df['CleanedReviewText'] = df['ReviewText'].astype(str).apply(lambda x: remove_stopwords_from_text(x, stopword_set))

from nltk.tokenize import word_tokenize
sample_text = df['CleanedReviewText'].iloc[0] if not df.empty else ""
tokens = word_tokenize(sample_text)
console.print(Panel(f"[bold cyan]Sample tokens:[/] {tokens}", title="Tokenization"))

tagged_tokens = nltk.pos_tag(tokens)
console.print(Panel(f"[bold cyan]POS Tagged tokens:[/] {tagged_tokens}", title="POS Tagging"))

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['PolarityScore'] = df['CleanedReviewText'].apply(lambda text: sia.polarity_scores(text)['compound'])
df['ReviewValue'] = df['Review'].map({'Recommended': 1, 'Not Recommended': 0})

correlation = df[['ReviewValue', 'PolarityScore']].corr(method='pearson')
console.print(Panel(f"[bold cyan]Pearson Correlation between ReviewValue and PolarityScore:[/]\n{correlation}", title="Correlation"))

# ---------------------------
# Extracting Most Used Positive and Negative Words
# ---------------------------
positive_lexicon = set(opinion_lexicon.positive())
negative_lexicon = set(opinion_lexicon.negative())

positive_counts = Counter()
negative_counts = Counter()

for review in df['CleanedReviewText']:
    tokens = set(word_tokenize(review.lower()))
    for token in tokens:
        if token in positive_lexicon:
            positive_counts[token] += 1
        if token in negative_lexicon:
            negative_counts[token] += 1

top_positive = positive_counts.most_common(10)
top_negative = negative_counts.most_common(10)

# Display top words in a Rich Table
table_pos = Table(title="Top 10 Positive Words by Review Count", style="bold green")
table_pos.add_column("Word", style="cyan", no_wrap=True)
table_pos.add_column("Review Count", style="magenta")
for word, count in top_positive:
    table_pos.add_row(word, str(count))
console.print(table_pos)

table_neg = Table(title="Top 10 Negative Words by Review Count", style="bold red")
table_neg.add_column("Word", style="cyan", no_wrap=True)
table_neg.add_column("Review Count", style="magenta")
for word, count in top_negative:
    table_neg.add_row(word, str(count))
console.print(table_neg)

if top_positive:
    pos_df = pd.DataFrame(top_positive, columns=['Word', 'Count'])
    fig_pos = px.bar(pos_df, x='Word', y='Count', title='Top 10 Positive Words by Review Count')
    fig_pos.show()

if top_negative:
    neg_df = pd.DataFrame(top_negative, columns=['Word', 'Count'])
    fig_neg = px.bar(neg_df, x='Word', y='Count', title='Top 10 Negative Words by Review Count')
    fig_neg.show()

recommendation_counts = df['Review'].value_counts().reset_index()
recommendation_counts.columns = ['Review', 'Count']
fig_bar = px.bar(recommendation_counts, x='Review', y='Count', title='Count of Recommendations')
fig_bar.show()

fig_box = px.box(df, x='Review', y='PolarityScore', title='Distribution of Polarity Scores by Review')
fig_box.show()

# Save the word cloud image inside the output folder
wordcloud_filename = os.path.join(output_folder, 'most-used')
plot_wordcloud(df['CleanedReviewText'], output_filename=wordcloud_filename)
