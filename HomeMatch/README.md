# Real Estate Listing Generator and Search System

This project provides a Python-based real estate listing generator and semantic search system. It leverages OpenAI's language model (LLM) to generate property descriptions and LangChain's vector embeddings to perform semantic searches based on user preferences.

## Features

- **Property Listing Generation**: Generate a real estate listing description based on neighborhood, price, bedrooms, bathrooms, and house size.
- **User Preference Parsing**: Parse user-provided preferences for desired features, amenities, and transportation options.
- **Vector Database Search**: Store and search real estate listings using semantic search in a vector database powered by ChromaDB.
- **Personalized Property Descriptions**: Customize a listing's description based on a user's preferences for important features, amenities, and urban living styles.

## Requirements

- Python 3.8+
- OpenAI API Key
- Required Libraries:
  - `openai`
  - `pandas`
  - `chromadb`
  - `langchain`
  - `re`

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/real-estate-listing.git
    cd real-estate-listing
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY="your-api-key"
    ```

## Usage

### 1. Generate Real Estate Listings

The script generates property listings based on a predefined list of neighborhoods and property details. The generated listings are saved to a CSV file.

### 2. Add Listings to Vector Database

The real estate listings are embedded into a vector database (ChromaDB) using OpenAI's embedding model. These listings can be searched later based on user preferences.

### 3. Query Listings with User Preferences

The system takes user inputs, parses preferences, and performs a semantic search against the vector database to find the best matching listings.

### 4. Personalize Listing Descriptions

Property descriptions can be personalized based on user preferences, such as the number of bedrooms, amenities, transportation needs, and more.

### Example Code Execution

To run the script, simply execute:

```bash
python main.py

## Files
- `main.py`: The main script to generate listings, store them in a vector database, and query listings based on user preferences.
- `listings.csv`' : The generated real estate listings stored in CSV format.

## License
This project is licensed under the MIT License.
Also, this project is the final project of Udacity Generative AI Nanodegree program.
General guidelines were given by Udacity.

