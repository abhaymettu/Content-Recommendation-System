from flask import Flask, request, jsonify, g, current_app
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import tensorflow as tf
import logging
from database import get_user_profile, get_content_metadata, store_user_profile, store_content_metadata, validate_user
from preprocessing import load_and_preprocess_data

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'super-secret-key'

jwt = JWTManager(app)
logging.basicConfig(level=logging.DEBUG)
model = tf.keras.models.load_model('recommendation_model.h5')
user_ids, content_ids, _ = load_and_preprocess_data()

@app.route('/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    if not validate_user(username, password):
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/recommend', methods=['POST'])
@jwt_required
def recommend():
    data = request.json
    user_id = data['user_id']

    scores = model.predict([user_id * len(content_ids), content_ids])
    recommended_content_ids = scores.argsort()[-10:][::-1]
    current_app.logger.info(f"Recommendations generated for user {user_id}")

    return jsonify({"recommended_contents": recommended_content_ids.tolist()})

@app.route('/user', methods=['POST', 'GET'])
@jwt_required
def user():
    if request.method == 'POST':
        data = request.json
        store_user_profile(data['user_id'], data['profile_data'])
        current_app.logger.info(f"Stored profile for user {data['user_id']}")
        return jsonify({"status": "success"})
    elif request.method == 'GET':
        user_id = request.args.get('user_id')
        profile = get_user_profile(user_id)
        return jsonify(profile)

@app.route('/content', methods=['POST', 'GET'])
@jwt_required
def content():
    if request.method == 'POST':
        data = request.json
        store_content_metadata(data['content_id'], data['metadata'])
        current_app.logger.info(f"Stored metadata for content {data['content_id']}")
        return jsonify({"status": "success"})
    elif request.method == 'GET':
        content_id = request.args.get('content_id')
        metadata = get_content_metadata(content_id)
        return jsonify(metadata)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found!"}), 404

@app.errorhandler(500)
def server_error(error):
    current_app.logger.error(f"Server error: {error}")
    return jsonify({"error": "Internal server error!"}), 500

if __name__ == '__main__':
    app.run(debug=True)
