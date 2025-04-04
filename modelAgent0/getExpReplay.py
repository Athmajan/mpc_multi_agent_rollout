from pymongo import MongoClient

def fetch_filtered_documents(collectionName):
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
