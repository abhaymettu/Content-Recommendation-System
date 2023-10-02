## Content Recommendation System

A sophisticated machine learning-based content recommendation system using collaborative filtering techniques with TensorFlow. The system offers personalized content suggestions for users and manages user profiles as well as content metadata in MongoDB. Additionally, it integrates user authentication for secure access.

### System Features:

1. **Deep Learning-Based Recommendations**: Uses a deep learning model with user and content embeddings to predict user ratings for content.
2. **Regularization & Dropout**: To prevent overfitting, the model has been designed with L2 regularization and dropout layers.
3. **Dynamic Learning Rate**: Employs an exponential decay schedule for adjusting the learning rate during training.
4. **User Authentication**: Integrated JWT-based user authentication for secure API access.
5. **Logging**: Incorporates logging for tracking server activity and potential issues.

### API Endpoints:

1. **Login**:
   - Path: `/login`
   - Method: POST
   - Data: `{ "username": "your_username", "password": "your_password" }`
   - Response: JWT access token if credentials are valid.

2. **Recommend**:
   - Path: `/recommend`
   - Method: POST
   - Headers: `Authorization: Bearer your_access_token`
   - Data: `{ "user_id": your_user_id }`
   - Response: List of recommended content IDs.

3. **User**:
   - Path: `/user`
   - Methods: POST (to store user profile), GET (to retrieve user profile)
   - Headers: `Authorization: Bearer your_access_token`
   - Data (for POST): `{ "user_id": your_user_id, "profile_data": your_profile_data }`
   - Response: User profile data or success status.

4. **Content**:
   - Path: `/content`
   - Methods: POST (to store content metadata), GET (to retrieve content metadata)
   - Headers: `Authorization: Bearer your_access_token`
   - Data (for POST): `{ "content_id": your_content_id, "metadata": your_metadata }`
   - Response: Content metadata or success status.
   - 
### License:

This project is open-source and available under the MIT License.
