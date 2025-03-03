import os
import json
import glob
import pandas as pd

def process_chatlog_file(filepath):
    """
    Load a chatlog file, extract customer messages, add a conversation_id (from the file name),
    and return a DataFrame.
    """
    # Extract conversation_id from the file name (strip the extension)
    conversation_id = os.path.splitext(os.path.basename(filepath))[0]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for msg in data.get("messages", []):
        # Filter out agent messages (assuming 'Agent' identifies agent messages)
        if msg.get("sender") != "Agent":
            # Perform basic cleaning: strip whitespace and lowercase
            text = msg.get("content", "").strip().lower()
            records.append({"conversation_id": conversation_id, "message": text})
    
    return pd.DataFrame(records)

def aggregate_chatlogs(path_pattern):
    """
    Process all chatlog files matching the glob pattern and aggregate them.
    """
    files = glob.glob(path_pattern)
    dfs = [process_chatlog_file(file) for file in files]
    master_df = pd.concat(dfs, ignore_index=True)
    return master_df

# Usage:
chatlogs_df = aggregate_chatlogs(r"C:\Users\alvin\OneDrive\Desktop\Sentiment Analysis Test\Data Sample\*.json")
chatlogs_df.to_csv("cleaned_chatlogs.csv", index=False)
print("Cleaned chat logs saved to 'cleaned_chatlogs.csv'.")
