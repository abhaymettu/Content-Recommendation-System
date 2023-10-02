from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['recommendation_db']

def store_user_profile(user_id, profile_data):
    db.user_profiles.insert_one({"user_id": user_id, "data": profile_data})

def get_user_profile(user_id):
    return db.user_profiles.find_one({"user_id": user_id})

def store_content_metadata(content_id, metadata):
    db.content_metadata.insert_one({"content_id": content_id, "metadata": metadata})

def get_content_metadata(content_id):
    return db.content_metadata.find_one({"content_id": content_id})
