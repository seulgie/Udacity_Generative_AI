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
    embedding = embedding_model.embed_documents([description])[0]
    collection.add(
        ids=[listing_id],
        embeddings=[embedding],
        metadatas=[metadata]
    )

# Function to parse preferences from answers
def parse_preferences(answers):
    preferences = {}

    # Extract the number of bedrooms from the house size answer
    size_pattern = r"(\d+)-?bedrooms?"
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
        query_text=preferences["important_features"],   # Query based on embedding
        n_results=5,   # Number of results to return
        where=query   # apply filters (bedrooms, amenities, etc.)
    )

    return results

def semantic_search_with_filters(preferences, collection, embedding_model):
    """
    Perform a semantic search on the vector database with additional filtering.
    
    :param preferences: The parsed user preferences.
    :param collection: The ChromaDB collection containing the listings and embeddings.
    :param embedding_model: The model used to generate embeddings for queries.
    :return: A list of matching listings.
    """
    
    # Construct the search query text for semantic search
    search_query = f"""
    I'm looking for a {preferences['bedrooms']}-bedroom house with features such as {', '.join(preferences['important_features'])}.
    It should include amenities like {', '.join(preferences['amenities'])}. 
    I'm particularly interested in a neighborhood that offers {preferences['urban_preference']} and good transportation options, including {', '.join(preferences['transportation'])}.
    """

    # Generate an embedding for the search query
    query_embedding = embedding_model.embed_query(search_query)

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
    if isinstance(response, dict) and 'text' in response:
        personalized_description = response['text'].strip()
    else:
        personalized_description = response.strip()  # Handles cases where response is directly text
    
    return personalized_description

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
    client = chromadb.Client()
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
        "A comfortable 3-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
    ]

    user_preferences = parse_preferences(answers)
    print("Parsed User Preferences:", user_preferences)

    '''sample output
    "Parsed User Preferences":{
        "bedrooms":3,
        "important_features":[
            "A quiet neighborhood",
            "good local schools",
            "and convenient shopping options."
            ],
        "amenities":[
            "A backyard for gardening",
            "a two-car garage",
            "and a modern",
            "energy-efficient heating system."
            ],
        "transportation":[
            "Easy access to a reliable bus line",
            "proximity to a major highway",
            "and bike-friendly roads."
            ],
        "urban_preference":"A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
        }
    '''

    # Query the vector database
    matching_listings = query_vector_database(user_preferences, collection)
    print("Matching Listings:", matching_listings)

    # Run the search with semantic filters
    matching_listings2 = semantic_search_with_filters(user_preferences, collection, embedding_model)

    # Print the results
    print("Matching Listings with Semantic Filters:", matching_listings2)

    '''example output
    Matching Listings:
    [
        {
            'id': 'listing_1',
            'metadata': {
                'neighborhood': 'Green Oaks',
                'price': 800000,
                'bedrooms': 3,
                'bathrooms': 2,
                'house_size': 2000
            },
            'description': 'A comfortable 3-bedroom home in a quiet, eco-friendly neighborhood with good schools and convenient shopping.'
        },
        # Additional matching listings
    ]   
    '''

    personalized_description = personalize_listing_description(matching_listings[0]['description'], user_preferences)
    print("Personalized Description:", personalized_description)

    '''
    Personalized Description:
    "Discover this perfect 3-bedroom retreat, ideally situated in a serene neighborhood 
    that aligns with your desire for a quiet and family-friendly environment. 
    This home features beautiful hardwood floors that complement the open-concept kitchen, 
    ideal for entertaining and everyday living. Enjoy the spacious backyard, 
    perfect for your gardening hobby, and the two-car garage that adds convenience. 
    The modern, energy-efficient heating system ensures comfort throughout the year. 
    Experience suburban tranquility with easy access to urban amenities, including nearby schools and shopping, 
    all while benefiting from convenient transportation options such as a reliable bus line and bike-friendly roads."
    '''

if __name__ == "__main__":
    main()
