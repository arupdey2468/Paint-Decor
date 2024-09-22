import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page you want to scrape
url = 'https://www.bergerpaints.com/colour/colour-catalogue'

# Send a request to the website and get the HTML content
response = requests.get(url)
if response.status_code == 200:
    print("Successfully fetched the webpage.")
else:
    print("Failed to fetch the webpage.")
    exit()

# Parse the webpage content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Create empty lists to store the extracted data
color_labels = []
hex_values = []

# Find all the div elements with class 'm7-color-card'
color_cards = soup.find_all('div', class_='ColorStyle_colorBox__ibNJr ')

# Loop through each div and extract the data-label and data-hex attributes
for card in color_cards:
    data_label = card.get('data-label')  # Get the 'data-label' attribute
    data_hex = card.get('data-hex')      # Get the 'data-hex' attribute
    
    # Append the extracted data to the lists
    if data_label and data_hex:
        color_labels.append(data_label)
        hex_values.append(data_hex)

# Create a DataFrame to store the extracted data
df = pd.DataFrame({
    'Color Label': color_labels,
    'Hex Value': hex_values
})

# Save the data into a CSV file
csv_file = 'bergerpaints.csv'
df.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}")

print("Scraping completed.")

