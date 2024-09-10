import os
from langchain.llms import OpenAI
import pandas as pd
import re
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma


os.environ["OPENAI_API_KEY"] = "voc-16036678971266772004727669e078c31a1e7.46148998"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

# Initialize the LLM
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)

""" test 
response = llm("What are some innovative features of a real estate app?")
print(response) # Print out the response to see the structure
"""

# Create a function to generate real estate listings
def generate_listing(neighborhood, price, bedrooms, bathrooms, house_size):
    prompt = f"""
    Generate a real estate listing with the following details:
    Neighborhood: {neighborhood}
    Price: ${price}
    Bedrooms: {bedrooms}
    Bathrooms: {bathrooms}
    House Size: {house_size} sqft
    
    Provide a property description and a neighborhood description.
    """
    return llm(prompt)

# List of neighborhoods and property details
listings_data = [
    {"neighborhood": "Green Oaks", "price": 800000, "bedrooms": 3, "bathrooms": 2, "house_size": 2000},
    {"neighborhood": "Sunnybrook", "price": 1200000, "bedrooms": 4, "bathrooms": 3, "house_size": 3500},
    {"neighborhood": "Maple Ridge", "price": 950000, "bedrooms": 4, "bathrooms": 2.5, "house_size": 2800},
    {"neighborhood": "Harbor View", "price": 600000, "bedrooms": 2, "bathrooms": 2, "house_size": 1500},
    {"neighborhood": "Cedar Hills", "price": 500000, "bedrooms": 3, "bathrooms": 1.5, "house_size": 1700},
    {"neighborhood": "Lakeside Estates", "price": 1500000, "bedrooms": 5, "bathrooms": 4, "house_size": 4200},
    {"neighborhood": "Willow Springs", "price": 780000, "bedrooms": 3, "bathrooms": 2, "house_size": 2100},
    {"neighborhood": "Riverstone", "price": 1250000, "bedrooms": 4, "bathrooms": 3.5, "house_size": 3300},
    {"neighborhood": "Mountain View", "price": 850000, "bedrooms": 3, "bathrooms": 2.5, "house_size": 2400},
    {"neighborhood": "Pine Grove", "price": 675000, "bedrooms": 3, "bathrooms": 2, "house_size": 1900},
]

# Store the listings
listings = []

# Generate listings for each property
for data in listings_data:
    neighborhood = data["neighborhood"]
    price = data["price"]
    bedrooms = data["bedrooms"]
    bathrooms = data["bathrooms"]
    house_size = data["house_size"]
    
    print(f"Generating listing for {neighborhood}...")
    listing_text = generate_listing(neighborhood, price, bedrooms, bathrooms, house_size)
    
    listings.append({
        "neighborhood": neighborhood,
        "price": price,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "house_size": house_size,
        "description": listing_text.strip()  # Clean up the generated text
    })

# Convert to a DataFrame to view results
df_listings = pd.DataFrame(listings)
print(df_listings)

# Optionally, save the listings to a CSV for future use
df_listings.to_csv("real_estate_listings.csv", index=False)

# Initialize ChromaDB client
client = chromadb.client()

# Initialize OpenAI embeddings model
embedding_model = OpenAIEmbeddings()

# Set up ChromaDB collection to store listings and embeddings
collection = client.create_collection("real_estate_listings")

# Load real estate listings (from the previously generated listings csv file)
listings_df = pd.read_csv("real_estate_listings.csv")

# Function to add listings to the ChromaDB collection
def add_listing_to_chromadb(collection, listing_id, description, metadata):
    # Generate embedding for the listing description
    embedding = embedding_model.embed([description])[0]

    # Add the embedding and metadata to ChromaDB collection
    collection.add(
        ids=[listing_id],           # Unique ID for the listing
        embeddings=[embedding],     # The embedding (vector representation)
        metadatas=[metadata]        # Metadata for easier querying (e.g., neighborhood, price)
    )

# Iterate through listings and store each in the vector database
for index, row in listings_df.iterrows():
    metadata = {
        "neighborhood": row["neighborhood"],
        "price": row["price"],
        "bedrooms": row["bedrooms"],
        "bathrooms": row["bathrooms"],
        "house_size": row["house_size"]
    }
    
    description = row["description"]
    listing_id = f"listing_{index}"  # Create a unique ID for each listing

    # Add the listing to the vector database
    add_listing_to_chromadb(collection, listing_id, description, metadata)

print("All listings have been stored in the vector database.")

# Retrieve a list of all stored listings
results = collection.get()
print("Stored listings in vector database:", results)

# Define buyer questions
questions = [
    "How big do you want your house to be?",
    "What are the 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?"
]

# Sample hard-coded answers (this can be interactive or collected from users via an input form)
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]

# Function to parse preferences from answers
def parse_preferences(answers):
    preferences = {}

    # Extract the number of bedrooms from the house size answer
    size_pattern = r"(\d+)-?bedroom"
    match = re.search(size_pattern, answers[0], re.IGNORECASE)
    if match:
        preferences["bedrooms"] = int(match.group(1))
    else:
        preferences["bedrooms"] = "Not specified"

    # Extract important features for the property
    preferences["important_features"] = answers[1].split(", ")

    # Extract amenities from the answers
    preferences["amenities"] = answers[2].split(", ")

    # Extract transportation preferences
    preferences["transportation"] = answers[3].split(", ")

    # Extract neighborhood preference
    preferences["urban_preference"] = answers[4]

    return preferences

# Parse the answers into structured preferences
user_preferences = parse_preferences(answers)
print("Parsed User Preferences:", user_preferences)


def query_vector_database(preferences, collection):
    # Build a basic query based on bedrooms and amenities (example)
    query = {
        "bedrooms": preferences["bedrooms"],
        "amenities": preferences["amenities"]
    }
    
    # Query the vector database
    results = collection.query(
        query_text=preferences["important_features"],  # Query based on embedding
        n_results=5,  # Number of results to return
        where=query  # Apply filters (bedrooms, amenities, etc.)
    )

    return results

# Simulate a query based on user preferences
matching_listings = query_vector_database(user_preferences, collection)
print("Matching Listings:", matching_listings)

def semantic_search_with_filters(preferences, collection, embedding_model):
    """
    Perform a semantic search on the vector database using buyer's preferences.
    
    :param preferences: The parsed user preferences in structured format.
    :param collection: The ChromaDB collection containing the listings and their embeddings.
    :param embedding_model: The model used to generate embeddings for queries.
    :return: A list of matching listings.
    """
    
    # Combine user preferences into a single query text for semantic search
    search_query = f"""
    I'm looking for a {preferences['bedrooms']}-bedroom house with features such as {', '.join(preferences['important_features'])}.
    It should include amenities like {', '.join(preferences['amenities'])}. 
    I'm particularly interested in a neighborhood that offers {preferences['urban_preference']} and good transportation options, including {', '.join(preferences['transportation'])}.
    """

    # Generate an embedding for the user's query
    query_embedding = embedding_model.embed([search_query])[0]

    # Define filters for metadata (e.g., bedrooms, bathrooms, price range)
    metadata_filters = {
        "bedrooms": preferences["bedrooms"]
    }

    # Perform the semantic search with metadata filters
    search_results = collection.query(
        query_embeddings=[query_embedding],   # Embedding for matching listings
        where=metadata_filters,               # Filter listings based on number of bedrooms, etc.
        n_results=5                           # Return top 5 matching listings
    )

    return search_results

def create_personalized_prompt(listing, preferences):
    """
    Create a prompt to personalize the property description.
    
    :param listing: A dictionary containing the property details and original description.
    :param preferences: The buyer's preferences.
    :return: A string prompt for the LLM.
    """
    
    original_description = listing['description']
    
    prompt = f"""
    Personalize the following property description to appeal to a buyer who is looking for a house with the following preferences:
    
    Preferences:
    - Number of Bedrooms: {preferences['bedrooms']}
    - Important Features: {', '.join(preferences['important_features'])}
    - Desired Amenities: {', '.join(preferences['amenities'])}
    - Transportation Needs: {', '.join(preferences['transportation'])}
    - Urban Preference: {preferences['urban_preference']}
    
    Original Description:
    {original_description}
    
    Personalized Description:
    """
    
    return prompt

def personalize_listing_description(listing, preferences):
    """
    Generate a personalized description for a property listing.
    
    :param listing: A dictionary containing the property details and original description.
    :param preferences: The buyer's preferences.
    :return: A personalized description of the property.
    """
    
    # Create a prompt for the LLM
    prompt = create_personalized_prompt(listing, preferences)
    
    # Generate personalized description using LLM
    response = llm(prompt)
    
    # Extract the personalized description from the LLM response
    personalized_description = response['text'].strip()
    
    return personalized_description

# Example usage with a sample listing and preferences
example_listing = {
    'description': 'This charming 3-bedroom home boasts beautiful hardwood floors, an open-concept kitchen, and a spacious backyard.'
}

example_preferences = {
    'bedrooms': 3,
    'important_features': ['quiet neighborhood', 'good local schools', 'convenient shopping options'],
    'amenities': ['backyard for gardening', 'two-car garage', 'energy-efficient heating system'],
    'transportation': ['easy access to bus line', 'proximity to major highway', 'bike-friendly roads'],
    'urban_preference': 'a balance between suburban tranquility and access to urban amenities'
}

personalized_description = personalize_listing_description(example_listing, example_preferences)
print("Personalized Description:", personalized_description)

