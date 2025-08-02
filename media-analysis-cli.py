# Import necessary libraries. You'll need to install these first!
# Open your terminal and run:
# pip install requests beautifulsoup4 textblob nltk
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
import re # For regular expressions, useful in bias detection
from urllib.parse import urljoin, urlparse # To handle links

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https') and bool(parsed.netloc)

class Article:
    """
    A class to store data about a news article, including its content and metadata.
    """
    def __init__(self, url):
        self.url = url
        self.text = ""
        self.title = ""
        self.authors = []
        self.external_links_count = 0

        # Call the method to fetch content as soon as an Article object is created.
        self.fetch_content()

    def fetch_content(self):
        """
        Fetches the article's text, title, and counts external links from the URL.
        """
        print(f"Fetching content from {self.url}...")
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # --- Extracting Title ---
            if soup.title and soup.title.string:
                self.title = soup.title.string.strip()
            else:
                self.title = "No title found"

            # --- Extracting Text and Links ---
            # We find the main content area to avoid counting nav/footer links.
            # Common tags for main content are <article>, <main>, or divs with specific IDs.

            article_body = soup.find('article') or soup.find('main') or soup
            
            paragraphs = article_body.find_all('p')
            self.text = "\n".join([p.get_text() for p in paragraphs])

            # Count external links within the article body as a proxy for sourcing
            links = set()
            base_url_netloc = urlparse(self.url).netloc
            for p in paragraphs:
                for a_tag in p.find_all('a', href=True):
                    href = a_tag['href']
                    absolute_url = urljoin(self.url, href)
                    # Check if it's a valid http/https URL and points to a different domain
                    if urlparse(absolute_url).scheme in ['http', 'https'] and urlparse(absolute_url).netloc != base_url_netloc:
                        links.add(absolute_url)
            
            self.external_links_count = len(links)

            if not self.text:
                print("Warning: Could not extract paragraph text. The report might be empty.")
                self.text = "Content could not be extracted."

            print("Fetching complete.")

        except requests.exceptions.RequestException as e:
            print(f"Error: Could not fetch the article. Please check the URL and your connection.")
            print(f"Details: {e}")
            retry = input("Would you like to try again? (y/n): ").lower()
            if retry == 'y':
                self.fetch_content()
            else:
                self.text = ""
                self.title = "Failed to Fetch"

class Analyzer:
    """
    The 'brains' of the operation. This class performs all the analysis on an article.
    """
    def __init__(self, article):
        if not isinstance(article, Article) or not article.text:
            raise ValueError("Analyzer requires a valid Article object with fetched text.")
        self.article = article
        self.text = article.text
        self.blob = TextBlob(self.text)

    def categorize_topic(self):
        """
        Categorizes the article's topic based on an expanded keyword analysis.
        """
        topics = {
            "Technology": ['ai', 'software', 'hardware', 'apple', 'google', 'data', 'cloud', 'startup', 'algorithm', 'cybersecurity', 'innovation', 'robotics', 'crypto','samsung','processor','graphics','performance'],
            "Politics": ['government', 'election', 'senate', 'law', 'policy', 'president', 'congress', 'political', 'legislation', 'democracy', 'ballot', 'vote'],
            "Sports": ['game', 'team', 'player', 'season', 'score', 'nba', 'football', 'olympics', 'champion', 'athlete', 'stadium', 'playoffs','goal','soccer'],
            "Business": ['company', 'market', 'stock', 'economy', 'ceo', 'finance', 'investment', 'revenue', 'quarterly', 'wall street', 'ipo', 'earnings'],
            "Health": ['health','sleep', 'medical', 'doctor', 'hospital', 'fda', 'virus', 'pandemic', 'vaccine', 'research', 'disease', 'wellness', 'nutrition'],
            "Entertainment": ['movie', 'music', 'singer', 'box office','celebrity', 'film', 'hollywood', 'nollywood', 'bollywood', 'award', 'series', 'netflix', 'actor', 'actress', 'director', 'album', 'concert'],
            "Education": ['course', 'student', 'test', 'exam', 'school', 'university', 'scholarship', 'hostel', 'bootcamp', 'camp']
        }
        
        scores = {topic: 0 for topic in topics}
        lower_text = self.text.lower()

        for topic, keywords in topics.items():
            for keyword in keywords:
                scores[topic] += lower_text.count(keyword)

        if all(score == 0 for score in scores.values()):
            return "Uncategorized"
            
        primary_topic = max(scores, key=scores.get)
        return primary_topic

    def analyze_sentiment(self):
        """
        Performs sentiment analysis using TextBlob.
        """
        polarity = self.blob.sentiment.polarity
        subjectivity = self.blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment_label = "Positive"
        elif polarity < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        return sentiment_label, polarity, subjectivity

    def detect_bias(self):
        """
        Detects potential bias by searching for a large list of emotionally charged or
        leading words and phrases.
        """
        biased_words = [
            'alarming', 'amazing', 'appalling', 'awful', 'bad', 'beautiful', 'best', 'blatant',
            'breakthrough', 'catastrophe', 'certainly', 'chaotic', 'clearly', 'collusion',
            'conspiracy', 'corrupt', 'covert', 'crisis', 'danger', 'deadly', 'decent',
            'definitely', 'disaster', 'disgraceful', 'disgusting', 'drastic', 'duty',
            'effective', 'excellent', 'exceptional', 'extreme', 'failure', 'fair',
            'fantastic', 'fear-mongering', 'finally', 'flawless', 'foolish', 'freedom',
            'frenzy', 'frightening', 'good', 'great', 'hate', 'healthy', 'heroic',
            'historic', 'honest', 'horrible', 'huge', 'immediately', 'important',
            'impossible', 'incompetent', 'incredible', 'inevitable', 'inflammatory',
            'injustice', 'inspirational', 'irresponsible', 'justice', 'landmark', 'likely',
            'looming', 'massive', 'masterpiece', 'meaningful', 'miracle', 'misleading',
            'monumental', 'must', 'myth', 'obviously', 'of course', 'outrageous', 'panic',
            'patriotic', 'perfect', 'pivotal', 'poor', 'propaganda', 'radical', 'reasonable',
            'revolutionary', 'rigged', 'scandal', 'scare', 'secret', 'shameful', 'shocking',
            'significant', 'smart', 'so-called', 'special', 'stupid', 'successful', 'suddenly',
            'superb', 'terrible', 'terrifying', 'threat', 'timely', 'tiny', 'tragic',
            'tremendous', 'true', 'trust', 'truth', 'unacceptable', 'unbelievable',
            'undoubtedly', 'unfair', 'unfortunate', 'unprecedented', 'urgent', 'victory',
            'violent', 'vital', 'wonderful', 'worst', 'wrong'
        ]
        
        found_words = []
        for word in biased_words:
            if re.search(r'\b' + re.escape(word) + r'\b', self.text, re.IGNORECASE):
                found_words.append(word)
        
        score = len(found_words)
        
        if score > 15:
            assessment = "High potential for bias"
        elif score > 7:
            assessment = "Moderate potential for bias"
        elif score > 0:
            assessment = "Low potential for bias"
        else:
            assessment = "Appears to be objective"
            
        return assessment, score, list(set(found_words))

    def analyze_source_credibility(self):
        """
        Analyzes source credibility based on subjectivity, article length, and the
        presence of external links (sourcing).
        """
        _sentiment_label, _polarity, subjectivity = self.analyze_sentiment()
        
        num_sentences = len(self.blob.sentences)
        link_count = self.article.external_links_count
        
        # Start with a score based on objectivity.
        credibility_score = (1 - subjectivity) * 100
        
        # Penalize very short articles that lack depth.
        if num_sentences < 8:
            credibility_score -= 25
        
        # Reward articles for citing external sources.
        # Add 3 points per link, up to a max bonus of 30.
        link_bonus = min(link_count * 3, 30)
        credibility_score += link_bonus
            
        # Clamp the score between 0 and 100.
        credibility_score = max(0, min(100, credibility_score))
        
        if credibility_score > 80:
            assessment = "Appears highly credible"
        elif credibility_score > 60:
            assessment = "Appears credible"
        elif credibility_score > 40:
            assessment = "Moderate credibility"
        else:
            assessment = "Low credibility (review with caution)"
            
        return assessment, f"{credibility_score:.2f}/100"


class ReportGenerator:
    """
    Generates and prints the final analysis report.
    """
    def __init__(self, article, analysis_results):
        self.article = article
        self.results = analysis_results

    def print_report(self):
        """Formats and prints the full report to the console."""
        print("\n" + "="*60)
        print(" " * 20 + "MEDIA ANALYSIS REPORT")
        print("="*60)
        print(f"Article Title: {self.article.title}")
        print(f"Source URL: {self.article.url}")
        print("-"*60)

        # --- Topic ---
        print(f"[*] Primary Topic: {self.results.get('topic', 'N/A')}")
        print("-" * 60)

        # --- Sentiment ---
        sentiment = self.results.get('sentiment', ('N/A', 0, 0))
        print(f"[*] Sentiment Analysis:")
        print(f"      - Overall Tone: {sentiment[0]}")
        print(f"      - Polarity Score: {sentiment[1]:.2f} (Negative to Positive, -1 to 1)")
        print(f"      - Subjectivity Score: {sentiment[2]:.2f} (Objective to Subjective, 0 to 1)")
        print("-" * 60)

        # --- Bias ---
        bias = self.results.get('bias', ('N/A', 0, []))
        print(f"[*] Bias Detection:")
        print(f"      - Assessment: {bias[0]}")
        print(f"      - Loaded Word Count: {bias[1]}")
        if bias[2]:
            # Show up to 5 found words to give a sample
            display_words = bias[2][:5]
            ellipsis = "..." if len(bias[2]) > 5 else ""
            print(f"      - Sample Words Found: {', '.join(display_words)}{ellipsis}")
        print("-" * 60)

        # --- Credibility ---
        credibility = self.results.get('credibility', ('N/A', 'N/A'))
        print(f"[*] Source Credibility Analysis:")
        print(f"      - Assessment: {credibility[0]}")
        print(f"      - Credibility Score: {credibility[1]}")
        print(f"      - External Links Found: {self.article.external_links_count} (Used in scoring)")
        print("="*60)
        print("Disclaimer: This is an automated analysis and should be used as a")
        print("guide, not a definitive judgment. Always read critically.")
        print("="*60 + "\n")


class CLI:
    """
    The main Command-Line Interface controller.
    """
    def run(self):
        """Starts the CLI application."""
        print("Welcome to the Media Analysis CLI!")
        
        url = input("Please enter the URL of a news article to analyze: ")

        if not is_valid_url(url):
            print("Error: Please enter a valid URL (e.g., https://...)")
            return
        # 1. Create an Article object (this also fetches the content)
        article = Article(url)

        if not article.text or article.title == "Failed to Fetch":
            print("Analysis cannot proceed. Exiting.")
            return

        # 2. Create an Analyzer object, passing the whole article
        analyzer = Analyzer(article)

        # 3. Run all analyses and store results in a dictionary
        analysis_results = {
            "topic": analyzer.categorize_topic(),
            "sentiment": analyzer.analyze_sentiment(),
            "bias": analyzer.detect_bias(),
            "credibility": analyzer.analyze_source_credibility()D
        }

        # 4. Generate and print the report
        report_generator = ReportGenerator(article, analysis_results)
        report_generator.print_report()


if __name__ == "__main__":
    try:
        cli = CLI()
        cli.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")