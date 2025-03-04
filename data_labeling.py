

import os
import time
import pandas as pd
import openai
from openai import OpenAIError  
from collections import Counter
from dotenv import load_dotenv

try:
    from openai.error import RateLimitError
except ModuleNotFoundError:
    RateLimitError = OpenAIError


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def label_message(message):
    max_retries = 3
    base_delay = 4  
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": (
                        "Label the following message as either 'Open', 'Neutral', or 'Not Open' "
                        "regarding openness to Christianity."
                    )},
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait_time = base_delay * (2 ** attempt)
            print(f"Rate limit reached while labeling message. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error labeling message: {e}")
            return "Error"
    return "Error"

def label_conversation(transcript):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": (
                        "Label the following conversation as either 'Open', 'Neutral', or 'Not Open' "
                        "regarding openness to Christianity."
                    )},
                    {"role": "user", "content": transcript}
                ]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limit reached while labeling conversation. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error labeling conversation: {e}")
            return "Error"
    return "Error"

def majority_vote(labels):
    """Return the label that accounts for more than 50% of the votes, or None if ambiguous."""
    if not labels:
        return None
    counts = Counter(labels)
    total = sum(counts.values())
    most_common, freq = counts.most_common(1)[0]
    if freq / total >= 0.5:
        return most_common
    return None


# Load cleaned CSV that contains 'conversation_id' and 'message' columns
df = pd.read_csv("cleaned_chatlogs.csv")

# Label each message with a delay to help manage rate limits
df['label'] = df['message'].apply(lambda m: (time.sleep(1) or label_message(m)))

# Save the message-level labeled data (optional)
df.to_csv("labeled_chatlogs.csv", index=False)
print("Message-level labeling complete and saved to 'labeled_chatlogs.csv'.")


# Group messages by conversation_id
grouped = df.groupby("conversation_id")
conversation_results = []

for conv_id, group in grouped:
    
    transcript = " ".join(group["message"].tolist())
 
    message_labels = group["label"].tolist()
    
    # Determine conversation sentiment via majority vote
    conv_label = majority_vote(message_labels)
    
    # If ambiguous, re-label using the full transcript
    if conv_label is None:
        conv_label = label_conversation(transcript)
    
    conversation_results.append({
        "conversation_id": conv_id,
        "transcript": transcript,
        "aggregated_sentiment": conv_label,
        "num_messages": len(group)
    })


conversation_df = pd.DataFrame(conversation_results)
conversation_df.to_csv("conversation_level_sentiment.csv", index=False)
print("Conversation-level sentiment saved to 'conversation_level_sentiment.csv'.")
