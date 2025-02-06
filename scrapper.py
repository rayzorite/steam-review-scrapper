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

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# =============================================================================
# SCRAPING MODULE
# =============================================================================
def init_browser(headless=True):
    """Initialize and return a headless Firefox browser."""
    options = Options()
    if headless:
        options.add_argument("--headless")
    return webdriver.Firefox(options=options)

def get_game_name(browser, game_id):
    """Fetch the game name from the Steam Community page using the App ID."""
    game_url = f"https://store.steampowered.com/app/{game_id}/"
    browser.get(game_url)
    try:
        game_name = browser.find_element(By.XPATH, '//div[@class="apphub_AppName"]').text
        return game_name
    except NoSuchElementException:
        console.print("[bold red]Could not retrieve the game name. Please check that the App ID is valid.[/]")
        exit(1)

def scroll_or_click(browser):
    """Attempt to click the 'Show More' button; if not found, scroll to the bottom."""
    try:
        show_more_button = browser.find_element(By.CLASS_NAME, "btn_green_white_innerfade")
        show_more_button.click()
        console.print("[bold magenta]Clicked 'Show More' button...[/]")
    except NoSuchElementException:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        console.print("[dim]Scrolled to bottom, loading more reviews...[/]")
    time.sleep(3)

def get_steam_id(card):
    """Extract the unique Steam ID from a review card, if available."""
    try:
        profile_url = card.find_element(By.XPATH, './/div[@class="apphub_friend_block"]/a').get_attribute('href')
        return profile_url.split('/')[-1]
    except Exception:
        return None

def scrape_review_data(card):
    """
    Extract and return review data from a review card.
    Returns a tuple: (review_text, recommendation, review_length, play_hours, date_posted)
    """
    date_posted = card.find_element(By.XPATH, './/div[@class="apphub_CardTextContent"]/div[@class="date_posted"]').text.strip()

    try:
        compensation_text = card.find_element(By.CLASS_NAME, "received_compensation").text.strip()
    except NoSuchElementException:
        compensation_text = ""

    card_text = card.find_element(By.CLASS_NAME, "apphub_CardTextContent").text.strip()
    # Remove unwanted parts
    for exclusion in [date_posted, compensation_text, "EARLY ACCESS REVIEW"]:
        card_text = card_text.replace(exclusion, "")
    review_text = card_text.replace("\n", " ").replace("\t", " ").strip()
    review_length = len(review_text.replace(" ", ""))

    thumb_text = card.find_element(By.XPATH, './/div[@class="reviewInfo"]/div[2]').text.strip()
    play_hours = card.find_element(By.XPATH, './/div[@class="reviewInfo"]/div[3]').text.strip()
    return review_text, thumb_text, review_length, play_hours, date_posted

def fetch_reviews_batch(browser, seen_steam_ids):
    """
    Click or scroll to load more reviews, then return a list of tuples (steam_id, review_data)
    from the latest batch of review cards.
    """
    scroll_or_click(browser)
    new_reviews = []
    review_cards = browser.find_elements(By.CLASS_NAME, 'apphub_Card')[-50:]
    for card in review_cards:
        steam_id = get_steam_id(card)
        if steam_id and steam_id in seen_steam_ids:
            continue
        review_data = scrape_review_data(card)
        new_reviews.append((steam_id, review_data))
    return new_reviews

def scrape_reviews(browser, mode, target_count):
    """
    Scrape reviews using either 'balanced' or 'random' mode.
    In balanced mode, attempts to get equal numbers of Recommended and Not Recommended reviews.
    Returns a list of review data tuples.
    """
    seen_steam_ids = set()
    reviews = []
    scroll_attempt = 0
    max_scroll_attempts = 4

    if mode == 'balanced':
        positive_reviews, negative_reviews = [], []
        target_per_category = target_count

        while (len(positive_reviews) < target_per_category or len(negative_reviews) < target_per_category) and scroll_attempt < max_scroll_attempts:
            prev_total = len(positive_reviews) + len(negative_reviews)
            for steam_id, review_data in fetch_reviews_batch(browser, seen_steam_ids):
                recommendation = review_data[1].strip()
                if recommendation == "Recommended" and len(positive_reviews) < target_per_category:
                    positive_reviews.append(review_data)
                    if steam_id:
                        seen_steam_ids.add(steam_id)
                elif recommendation == "Not Recommended" and len(negative_reviews) < target_per_category:
                    negative_reviews.append(review_data)
                    if steam_id:
                        seen_steam_ids.add(steam_id)
            current_total = len(positive_reviews) + len(negative_reviews)
            if current_total == prev_total:
                scroll_attempt += 1
                console.print(f"[bold red]No new reviews found. Scroll attempt {scroll_attempt}/{max_scroll_attempts}.[/]")
            else:
                scroll_attempt = 0
            time.sleep(2)
        reviews = positive_reviews + negative_reviews

    else:  # Random mode
        while len(reviews) < target_count and scroll_attempt < max_scroll_attempts:
            prev_total = len(reviews)
            for steam_id, review_data in fetch_reviews_batch(browser, seen_steam_ids):
                reviews.append(review_data)
                if steam_id:
                    seen_steam_ids.add(steam_id)
                if len(reviews) >= target_count:
                    break
            if len(reviews) == prev_total:
                scroll_attempt += 1
                console.print(f"[bold red]No new reviews found. Scroll attempt {scroll_attempt}/{max_scroll_attempts}.[/]")
            else:
                scroll_attempt = 0
            time.sleep(2)

    return reviews

# =============================================================================
# DATA PROCESSING MODULE
# =============================================================================
def process_data(reviews, output_year='2024'):
    """
    Create a DataFrame from scraped reviews and clean the data.
    Returns the cleaned DataFrame.
    """
    columns = ['ReviewText', 'Review', 'ReviewLength', 'PlayHours', 'DatePosted']
    df = pd.DataFrame(reviews, columns=columns)

    # Clean PlayHours
    df['PlayHours'] = df['PlayHours'].str.replace("hrs on record", "", regex=False).str.strip()

    # Standardize DatePosted
    month_mapping = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04',
        'May': '05', 'June': '06', 'July': '07', 'August': '08',
        'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }
    date_extracted = df['DatePosted'].str.extract(r'(\d+)\s+(\w+)', expand=True)
    df['Day'] = date_extracted[0]
    df['Month'] = date_extracted[1].map(month_mapping)
    df['DatePosted'] = pd.to_datetime(df['Day'] + '/' + df['Month'] + f'/{output_year}', format='%d/%m/%Y')
    df['DatePosted'] = df['DatePosted'].dt.strftime('%d-%m-%Y')
    df.drop(['Day', 'Month'], axis=1, inplace=True)

    df['ReviewText'] = df['ReviewText'].str.strip()
    df['Review'] = df['Review'].str.strip()

    return df

def save_data(df, output_folder="result"):
    """
    Save the DataFrame to CSV and JSON files inside the specified output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    csv_filename = os.path.join(output_folder, 'reviews.csv')
    df.to_csv(csv_filename, encoding='utf-8', sep=';', index=False)
    console.print(Panel(f"[bold green]Cleaned CSV data saved to:[/] {csv_filename}", title="Created CSV"))

    json_filename = os.path.join(output_folder, 'reviews.json')
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(df.to_dict(orient='records'), json_file, ensure_ascii=False, indent=4)
    console.print(Panel(f"[bold green]Cleaned JSON data saved to:[/] {json_filename}", title="Created JSON"))

# =============================================================================
# TEXT ANALYSIS & VISUALIZATION MODULE
# =============================================================================
def remove_stopwords_from_text(text, stopword_set):
    """Remove stopwords from text and return the cleaned string."""
    return ' '.join(word for word in text.split() if word.lower() not in stopword_set)

def analyze_text(df):
    """
    Perform text tokenization, POS tagging, sentiment analysis and extract top positive/negative words.
    Returns the updated DataFrame and top word counts.
    """
    stopword_set = set(nltk_stopwords.words('english'))
    df['CleanedReviewText'] = df['ReviewText'].astype(str).apply(lambda x: remove_stopwords_from_text(x, stopword_set))

    # Tokenization and POS tagging for a sample
    from nltk.tokenize import word_tokenize
    sample_text = df['CleanedReviewText'].iloc[0] if not df.empty else ""
    tokens = word_tokenize(sample_text)
    console.print(Panel(f"[bold cyan]Sample tokens:[/] {tokens}", title="Tokenization"))
    tagged_tokens = nltk.pos_tag(tokens)
    console.print(Panel(f"[bold cyan]POS Tagged tokens:[/] {tagged_tokens}", title="POS Tagging"))

    # Sentiment analysis
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    df['PolarityScore'] = df['CleanedReviewText'].apply(lambda text: sia.polarity_scores(text)['compound'])
    df['ReviewValue'] = df['Review'].map({'Recommended': 1, 'Not Recommended': 0})
    correlation = df[['ReviewValue', 'PolarityScore']].corr(method='pearson')
    console.print(Panel(f"[bold cyan]Pearson Correlation:[/]\n{correlation}", title="Correlation"))

    # Count positive/negative words
    positive_lexicon = set(opinion_lexicon.positive())
    negative_lexicon = set(opinion_lexicon.negative())
    positive_counts = Counter()
    negative_counts = Counter()

    from nltk.tokenize import word_tokenize
    for review in df['CleanedReviewText']:
        tokens = set(word_tokenize(review.lower()))
        for token in tokens:
            if token in positive_lexicon:
                positive_counts[token] += 1
            if token in negative_lexicon:
                negative_counts[token] += 1

    top_positive = positive_counts.most_common(10)
    top_negative = negative_counts.most_common(10)

    # Display top words using Rich Tables
    table_pos = Table(title="Top 10 Positive Words", style="bold green")
    table_pos.add_column("Word", style="cyan", no_wrap=True)
    table_pos.add_column("Count", style="magenta")
    for word, count in top_positive:
        table_pos.add_row(word, str(count))
    console.print(table_pos)

    table_neg = Table(title="Top 10 Negative Words", style="bold red")
    table_neg.add_column("Word", style="cyan", no_wrap=True)
    table_neg.add_column("Count", style="magenta")
    for word, count in top_negative:
        table_neg.add_row(word, str(count))
    console.print(table_neg)

    # Plot visualizations using Plotly
    if top_positive:
        pos_df = pd.DataFrame(top_positive, columns=['Word', 'Count'])
        fig_pos = px.bar(pos_df, x='Word', y='Count', title='Top 10 Positive Words')
        fig_pos.show()

    if top_negative:
        neg_df = pd.DataFrame(top_negative, columns=['Word', 'Count'])
        fig_neg = px.bar(neg_df, x='Word', y='Count', title='Top 10 Negative Words')
        fig_neg.show()

    recommendation_counts = df['Review'].value_counts().reset_index()
    recommendation_counts.columns = ['Review', 'Count']
    fig_bar = px.bar(recommendation_counts, x='Review', y='Count', title='Count of Recommendations')
    fig_bar.show()

    fig_box = px.box(df, x='Review', y='PolarityScore', title='Polarity Score Distribution by Review')
    fig_box.show()

    return df

def plot_wordcloud(series, output_filename='wordcloud'):
    """Generate, save, and display a word cloud image from a pandas Series."""
    text = ' '.join(series.astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    file_path = f"{output_filename}.png"
    wordcloud.to_file(file_path)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # ---------------------------
    # User Input & Setup
    # ---------------------------
    app_id_input = questionary.text("Enter Game's App ID:").ask()
    try:
        GAME_ID = int(app_id_input)
    except ValueError:
        console.print("[bold red]Invalid App ID. Please enter a numeric App ID.[/]")
        exit(1)

    scrape_mode = questionary.select(
        "Choose scraping mode:",
        choices=["balanced", "random"]
    ).ask()

    reviews_count_input = questionary.text("Enter the number of reviews to scrape (Default 20 Reviews):").ask()
    try:
        target_reviews = int(reviews_count_input)
    except ValueError:
        console.print("[bold yellow]Invalid number entered. Defaulting to 20 reviews.[/]")
        target_reviews = 20

    # Initialize browser and download NLTK resources
    browser = init_browser(headless=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('opinion_lexicon', quiet=True)

    # Get game name and open the review page
    game_name = get_game_name(browser, GAME_ID)
    url = f"https://steamcommunity.com/app/{GAME_ID}/reviews/?p=1&browsefilter=mostrecent&filterLanguage=english"
    console.print(Panel(f"[bold cyan]URL:[/] {url}\n[bold cyan]Game: {game_name}[/]", title="Game Information"))
    browser.get(url)

    # Create output folder
    output_folder = "result"
    os.makedirs(output_folder, exist_ok=True)
    console.print(Panel(f"[bold green]Output folder ready:[/] {output_folder}", title="Folder Setup"))

    # ---------------------------
    # Scraping Reviews
    # ---------------------------
    start_time = time.time()
    reviews = scrape_reviews(browser, scrape_mode, target_reviews)
    browser.quit()
    elapsed_time = time.time() - start_time
    mode_msg = (f"Positive: {len([r for r in reviews if r[1].strip() == 'Recommended'])}, "
                f"Negative: {len([r for r in reviews if r[1].strip() == 'Not Recommended'])}") if scrape_mode == 'balanced' \
                else f"Total: {len(reviews)}"
    console.print(Panel(f"[bold green]Scraping completed in:[/] {elapsed_time:.2f}s\n[bold green]Reviews scraped:[/] {mode_msg}",
                        title="Scraping Results"))

    # ---------------------------
    # Data Processing & Saving
    # ---------------------------
    df = process_data(reviews)
    save_data(df, output_folder)

    # ---------------------------
    # Text Analysis & Visualization
    # ---------------------------
    df = analyze_text(df)
    wordcloud_filename = os.path.join(output_folder, 'most-used')
    plot_wordcloud(df['CleanedReviewText'], output_filename=wordcloud_filename)

if __name__ == '__main__':
    main()
