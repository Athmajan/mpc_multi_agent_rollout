from pymongo import MongoClient

def fetch_filtered_documents(collectionName):
    """
    Fetch documents from the specified MongoDB collection where the maximum 'step'
    for each 'seed' is less than 34, and return the previous observations and actions
    for agent_0 for those documents.

    The function performs the following steps:
    - Connect to MongoDB at 'mongodb://localhost:27017/'.
    - Use the database named 'mpe_continuous'.
    - Aggregate the collection to find the maximum 'step' for each distinct 'seed'.
    - Select seeds where their maximum 'step' is less than 34.
    - Query the collection for documents matching those seeds and project the
      'prev_observations', 'actions', and 'seed' fields.
    - Extract the 'agent_0' entries from 'prev_observations' and 'actions' for each result
      and return a list of dictionaries with keys: 'seed', 'prev_observations', 'actions'.

    Parameters:
    - collectionName (str): Name of the collection within the 'mpe_continuous' database
      to query.

    Returns:
    - list[dict]: A list of dictionaries, each containing:
        - 'seed': the seed identifier for the document
        - 'prev_observations': the 'agent_0' entry from the document's 'prev_observations' field
        - 'actions': the 'agent_0' entry from the document's 'actions' field

    Raises:
    - pymongo.errors.PyMongoError (or subclass): If there is an error connecting to MongoDB
      or running the aggregation/query.
    - KeyError/TypeError: If expected nested keys (like 'agent_0') are missing or not shaped
      as expected in returned documents.

    Example:
    >>> documents = fetch_filtered_documents('seq_roll_2_4_10_20_3')
    >>> documents[0]
    {'seed': 123, 'prev_observations': {...}, 'actions': {...}}
    """
     

    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB URI if different
    db = client['mpe_continuous']  # Replace with your database name
    # collection = db['seq_roll_2_4_10_20_3']  # Replace with your collection name
    collection = db[collectionName]  # Replace with your collection name

    # Query to find maximum "step" per "seed"
    pipeline = [
        {"$group": {"_id": "$seed", "max_step": {"$max": "$step"}}},
        {"$match": {"max_step": {"$lt": 34}}}
    ]

    # Fetch seeds where the maximum step is less than 34
    seeds_with_low_steps = list(collection.aggregate(pipeline))
    seeds = [doc['_id'] for doc in seeds_with_low_steps]

    # Retrieve previous observations and actions for the matching seeds
    results = collection.find(
        {"seed": {"$in": seeds}},
        {"prev_observations": 1, "actions": 1,"seed":1}
    )

    # import ipdb; ipdb.set_trace()

    # Convert results to a list of dictionaries
    output = [{"seed": doc.get("seed"),"prev_observations": doc.get("prev_observations")["agent_0"], "actions": doc.get("actions")["agent_0"]} for doc in results]

    return output

if __name__ == "__main__":
    documents = fetch_filtered_documents('seq_roll_2_4_10_20_3')
    for doc in documents:
        print(doc)
        break
