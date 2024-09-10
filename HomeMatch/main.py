import os
import re
import pandas as pd
import chromadb
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

# Set environment variables for OpenAI API
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Initialize the OpenAI LLM
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)

def generate_listing(neighborhood, price, bedrooms, bathrooms, house_size):
    """
    Generate a real estate listing based on provided details.

    :param neighborhood: Neighborhood name.
    :param price: Price of the property.
    :param bedrooms: Number of bedrooms.
    :param bathrooms: Number of bathrooms.
    :param house_size: Size of the house in sqft.
    :return: Generated property description and neighborhood description.
    """
    prompt = (f"Generate a real estate listing with the following details:\n"
              f"Neighborhood: {neighborhood}\n"
              f"Price: ${price}\n"
              f"Bedrooms: {bedrooms}\n"
              f"Bathrooms: {bathrooms}\n"
              f"House Size: {house_size} sqft\n\n"
              "Provide a property description and a neighborhood description.")
    return llm(prompt).strip()

def add_listing_to_chromadb(collection, listing_id, description, metadata, embedding_model):
    """
    Add a listing to the ChromaDB collection.

    :param collection: ChromaDB collection instance.
    :param listing_id: Unique ID for the listing.
    :param description: Description of the property.
    :param metadata: Metadata related to the property.
    :param embedding_model: Model used for generating embeddings.
    """
    embedding = embedding_model.embed([description])[0]
    collection.add(
        ids=[listing_id],
        embeddings=[embedding],
        metadatas=[metadata]
    )

def parse_preferences(answers):
    """
    Parse user preferences from answers.

    :param answers: List of answers from the user.
    :return: Dictionary containing parsed preferences.
    """
    preferences = {}

    size_pattern = r"(\d+)-?bedroom"
    match = re.search(size_pattern, answers[0], re.IGNORECASE)
    preferences["bedrooms"] = int(match.group(1)) if match else "Not specified"

    preferences["important_features"] = answers[1].split(", ")
    preferences["amenities"] = answers[2].split(", ")
    preferences["transportation"] = answers[3].split(", ")
    preferences["urban_preference"] = answers[4]

    return preferences

def query_vector_database(preferences, collection):
    """
    Query the vector database based on user preferences.

    :param preferences: User preferences.
    :param collection: ChromaDB collection instance.
    :return: List of matching listings.
    """
    query = {
        "bedrooms": preferences["bedrooms"],
        "amenities": preferences["amenities"]
    }

    results = collection.query(
        query_text=preferences["important_features"],
        n_results=5,
        where=query
    )

    return results

def semantic_search_with_filters(preferences, collection, embedding_model):
    """
    Perform a semantic search on the vector database with filters.

    :param preferences: Parsed user preferences.
    :param collection: ChromaDB collection instance.
    :param embedding_model: Model used for generating query embeddings.
    :return: List of matching listings.
    """
    search_query = (f"I'm looking for a {preferences['bedrooms']}-bedroom house with features such as "
                    f"{', '.join(preferences['important_features'])}. It should include amenities like "
                    f"{', '.join(preferences['amenities'])}. I'm particularly interested in a neighborhood "
                    f"that offers {preferences['urban_preference']} and good transportation options, including "
                    f"{', '.join(preferences['transportation'])}.")

    query_embedding = embedding_model.embed([search_query])[0]

    metadata_filters = {
        "bedrooms": preferences["bedrooms"]
    }

    search_results = collection.query(
        query_embeddings=[query_embedding],
        where=metadata_filters,
        n_results=5
    )

    return search_results

def create_personalized_prompt(listing, preferences):
    """
    Create a personalized prompt for the property description.

    :param listing: Dictionary containing property details and description.
    :param preferences: User preferences.
    :return: String prompt for the LLM.
    """
    original_description = listing['description']

    prompt = (f"Personalize the following property description to appeal to a buyer who is looking for a house "
              f"with the following preferences:\n\n"
              f"Preferences:\n"
              f"- Number of Bedrooms: {preferences['bedrooms']}\n"
              f"- Important Features: {', '.join(preferences['important_features'])}\n"
              f"- Desired Amenities: {', '.join(preferences['amenities'])}\n"
              f"- Transportation Needs: {', '.join(preferences['transportation'])}\n"
              f"- Urban Preference: {preferences['urban_preference']}\n\n"
              f"Original Description:\n{original_description}\n\n"
              f"Personalized Description:\n")
    
    return prompt

def personalize_listing_description(listing, preferences):
    """
    Generate a personalized description for a property listing.

    :param listing: Dictionary containing property details and description.
    :param preferences: User preferences.
    :return: Personalized property description.
    """
    prompt = create_personalized_prompt(listing, preferences)
    response = llm(prompt)
    return response.strip()

def main():
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

    # Generate listings for each property
    listings = []
    for data in listings_data:
        print(f"Generating listing for {data['neighborhood']}...")
        listing_text = generate_listing(
            data["neighborhood"],
            data["price"],
            data["bedrooms"],
            data["bathrooms"],
            data["house_size"]
        )
        listings.append({
            "neighborhood": data["neighborhood"],
            "price": data["price"],
            "bedrooms": data["bedrooms"],
            "bathrooms": data["bathrooms"],
            "house_size": data["house_size"],
            "description": listing_text
        })

    # Convert to a DataFrame and save to CSV
    df_listings = pd.DataFrame(listings)
    df_listings.to_csv("listings.csv", index=False)

    # Initialize ChromaDB client and embedding model
    client = chromadb.client()
    embedding_model = OpenAIEmbeddings()
    collection = client.create_collection("real_estate_listings")

    # Load real estate listings from CSV file
    listings_df = pd.read_csv("listings.csv")

    # Add listings to the vector database
    for index, row in listings_df.iterrows():
        metadata = {
            "neighborhood": row["neighborhood"],
            "price": row["price"],
            "bedrooms": row["bedrooms"],
            "bathrooms": row["bathrooms"],
            "house_size": row["house_size"]
        }
        description = row["description"]
        listing_id = f"listing_{index}"
        add_listing_to_chromadb(collection, listing_id, description, metadata, embedding_model)

    print("All listings have been stored in the vector database.")

    # Define and parse user preferences
    answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
    ]
    user_preferences = parse_preferences(answers)
    print("Parsed User Preferences:", user_preferences)

    # Query the vector database
    matching_listings = query_vector_database(user_preferences, collection)
    print("Matching Listings:", matching_listings)

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

if __name__ == "__main__":
    main()
