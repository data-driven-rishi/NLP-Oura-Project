import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
from collections import Counter
import re

# Download stopwords (one time)
nltk.download('stopwords')

# Load your CSV file
df = pd.read_csv('data/NLP.csv')

# Remove rows where Experience is empty
df_clean = df.dropna(subset=['Experience'])

# Function to get sentiment
def get_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df_clean['Sentiment'] = df_clean['Experience'].apply(get_sentiment)

# Function to extract words
def get_words(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Split into words
    words = text.split()
    # Remove stopwords (common words like 'the', 'a', 'is')
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words

# Get words for each sentiment
positive_words = []
negative_words = []
neutral_words = []

for idx, row in df_clean.iterrows():
    words = get_words(row['Experience'])
    if row['Sentiment'] == 'Positive':
        positive_words.extend(words)
    elif row['Sentiment'] == 'Negative':
        negative_words.extend(words)
    else:
        neutral_words.extend(words)

# Count most common words
print("=== TOP 15 WORDS IN POSITIVE EXPERIENCES ===")
positive_counter = Counter(positive_words)
for word, count in positive_counter.most_common(15):
    print(f"{word}: {count}")

print("\n=== TOP 15 WORDS IN NEGATIVE EXPERIENCES ===")
negative_counter = Counter(negative_words)
for word, count in negative_counter.most_common(15):
    print(f"{word}: {count}")

print("\n=== TOP 15 WORDS IN NEUTRAL EXPERIENCES ===")
neutral_counter = Counter(neutral_words)
for word, count in neutral_counter.most_common(15):
    print(f"{word}: {count}")

import matplotlib.pyplot as plt

# Visualization 1: Sentiment Breakdown (Pie Chart)
sentiment_counts = df_clean['Sentiment'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'gray', 'red'])
plt.title('Sentiment Breakdown of Experiences')
plt.tight_layout()
plt.savefig('sentiment_pie_chart.png')
print("✓ Saved: sentiment_pie_chart.png")
plt.close()

# Visualization 2: Top Words in Positive Experiences (Bar Chart)
plt.figure(figsize=(12, 6))
top_positive = positive_counter.most_common(10)
words, counts = zip(*top_positive)
plt.bar(words, counts, color='green')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Words in Positive Experiences')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('positive_words.png')
print("✓ Saved: positive_words.png")
plt.close()

# Visualization 3: Top Words in Negative Experiences (Bar Chart)
plt.figure(figsize=(12, 6))
top_negative = negative_counter.most_common(10)
words, counts = zip(*top_negative)
plt.bar(words, counts, color='red')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Words in Negative Experiences')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('negative_words.png')
print("✓ Saved: negative_words.png")
plt.close()

# Visualization 4: Top Words in Neutral Experiences (Bar Chart)
plt.figure(figsize=(12, 6))
top_neutral = neutral_counter.most_common(10)
words, counts = zip(*top_neutral)
plt.bar(words, counts, color='gray')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Words in Neutral Experiences')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('neutral_words.png')
print("✓ Saved: neutral_words.png")
plt.close()

print("\n✓ All visualizations saved!") 

# Show negative experiences
print("\n=== THE 2 NEGATIVE EXPERIENCES ===\n")
negative_experiences = df_clean[df_clean['Sentiment'] == 'Negative']
for idx, row in negative_experiences.iterrows():
    print(f"Participant #{row['Participant Number']}")
    print(f"Experience: {row['Experience']}")
    print("-" * 80)


