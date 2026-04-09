import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_leases():
    # Example URL (Note: many legal sites have blocks, this is a generic structure)
    url = "https://www.lawinsider.com/clause/lease-agreement"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Scraping snippet texts
        clauses = [div.get_text().strip() for div in soup.find_all('div', class_='snippet-content')]
        
        # Save to data folder
        df = pd.DataFrame(clauses, columns=["legal_clause"])
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/raw_clauses.csv", index=False)
        print("Scraped successfully! Check data/raw_clauses.csv")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    scrape_leases()