# E-commerce Recommendation Microservice

This microservice provides personalized product recommendations to users based on their browsing and purchase history.

## Getting Started

### Prerequisites
docker-compose build to build the container
and 
docker-compose up to make it up and running and you can try the api endpoints at 

http://localhost:8000/docs
- Docker
- Docker Compose

### Build and Run

1. Clone the repository:

   ```bash
   git clone https://github.com/findsah/recommendation-engine-microservice.git
   cd recommendation-engine-microservice
The API will be accessible at http://localhost:8000.

API Documentation
The API documentation is available at http://localhost:8000/docs.

Endpoints
POST /recommend

Get personalized product recommendations for a user.

Request Body:

json
Copy code
{
  "user_id": 1
}

]
Configuration
The SQLite database is used for data storage.
Collaborative filtering is employed for recommendation.
The system is designed to be horizontally scalable.