import json
import re
import ollama
import os

def clean_ai_response(response_text):
    """Cleans the AI response to extract a valid JSON list."""
    # Remove backticks and language identifiers like ```json or '''
    cleaned_text = re.sub(r"```json|```|'''", "", response_text.strip())
    
    # Try to locate a JSON array within the cleaned text
    match = re.search(r'\[.*?\]', cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))  # Return as JSON list if valid
        except json.JSONDecodeError:
            print("Error: Cleaned AI-generated output was not in valid JSON format.")
    return None

def get_last_50_usernames(dataset):
    """Retrieve the last 50 usernames to use as part of the prompt."""
    last_50_usernames = dataset[-50:] if len(dataset) >= 50 else dataset
    return [entry['username'] for entry in last_50_usernames]

def generate_username_batch(last_50_usernames):
    """Generate a batch of 10 unique usernames using AI."""
    last_50_str = json.dumps(last_50_usernames, indent=4)
    
    prompt = f"""
    Using the following usernames as examples, create 10 unique usernames with varying patterns. Avoid replicating or following obvious patterns from the examples:
    {last_50_str}
    
    Examples of new usernames format:
    {{"username": "sunny123_run"}},
    {{"username": "sunny123_rainy"}},
    {{"username": "sunny123_snowflake"}},
    {{"username": "sunny123_glowing"}},
    """
    
    conversation_history = [
        {'role': 'user', 'content': 'Generate 10 random usernames in JSON format as described:' + prompt}
    ]
    
    response = ollama.chat(model='llama3.1', messages=conversation_history, stream=True)
    
    entry_text = ''
    for part in response:
        entry_text += part['message']['content']
    
    # Clean the response and parse it as JSON
    usernames = clean_ai_response(entry_text)
    if usernames:
        # Verify each entry has the required 'username' field
        valid_usernames = []
        for entry in usernames:
            if isinstance(entry, dict) and "username" in entry:
                valid_usernames.append(entry)
        
        return valid_usernames if len(valid_usernames) == 10 else None  # Ensure 10 valid usernames are returned
    else:
        print("Error: Failed to clean and parse AI response.")
        return None

def generate_usernames_dataset(total_entries, filename="username_dataset.json"):
    """Generate a dataset of unique usernames, appending to JSON file and avoiding duplicates."""
    unique_usernames = set()  # Track unique usernames
    dataset = []

    # Load existing usernames if the file already exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
            for entry in existing_data:
                unique_usernames.add(entry["username"])  # Add existing usernames to the set
            dataset.extend(existing_data)
    
    generated_count = len(dataset)

    while generated_count < total_entries:
        last_50_usernames = get_last_50_usernames(dataset)  # Get the latest 50 usernames for context
        batch = generate_username_batch(last_50_usernames)
        
        if batch is None:
            continue  # Retry if batch generation fails or is incomplete
        
        # Check each username in the batch for uniqueness
        for entry in batch:
            username = entry["username"]
            if username not in unique_usernames and generated_count < total_entries:
                unique_usernames.add(username)
                dataset.append(entry)
                generated_count += 1

        print(f"Generated {generated_count} of {total_entries} usernames\n")
        
    # Save the entire dataset to JSON file, appending to existing data
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Dataset generated and saved to {filename}")
    return dataset

# User prompts for dataset generation
total_entries = int(input("Enter the total number of usernames to generate: "))

# Generate the dataset and append to file
generate_usernames_dataset(total_entries)
