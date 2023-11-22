# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from fastapi.security import OAuth2PasswordBearer
from typing import List
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

# SQLAlchemy models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    location = Column(String)
    preferences = Column(String)

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String)
    name = Column(String)
    description = Column(String)
    tags = Column(String)

class PurchaseHistory(Base):
    __tablename__ = "purchase_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    rating = Column(Float)

# SQLite database connection
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initializing the database tables
Base.metadata.create_all(bind=engine)

# Load data into the database when the application starts
def load_data_into_db():
    df = pd.read_csv("dataset.csv")

    # Removing duplicates based on user_id
    df = df.drop_duplicates(subset=['user_id'])

    # Inserting users
    users_data = df[['user_id', 'name', 'age', 'gender', 'location', 'preferences']].drop_duplicates()
    users_data.rename(columns={'user_id': 'id'}, inplace=True)
    users_data.to_sql('users', con=engine, index=False, if_exists='replace')

    # Inserting products
    products_data = df[['product_id', 'category', 'product_name', 'description', 'tags']].drop_duplicates()
    products_data.rename(columns={'product_id': 'id', 'product_name': 'name'}, inplace=True)
    products_data.to_sql('products', con=engine, index=False, if_exists='replace')

    # Inserting transactions
    transactions_data = df[['user_id', 'product_id']]
    transactions_data['rating'] = 1  # Add a dummy 'rating' column
    transactions_data.to_sql('purchase_history', con=engine, index=False, if_exists='replace')

# Loading data into the database when the application starts
load_data_into_db()

# Loading data for collaborative filtering
reader = Reader(rating_scale=(1, 5))
transactions_data = pd.read_sql_table('purchase_history', con=engine)
data = Dataset.load_from_df(transactions_data[['user_id', 'product_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = KNNBasic(sim_options={'user_based': True})  # Set 'user_based' to True for user-based collaborative filtering
model.fit(trainset)

# FastAPI instance
app = FastAPI()

class RecommendationRequest(BaseModel):
    user_id: int

class RecommendationResponse(BaseModel):
    product_id: int
    score: float

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get the collaborative filtering model
def get_collaborative_filtering_model():
    reader = Reader(rating_scale=(1, 5))
    transactions_data = pd.read_sql_table('purchase_history', con=engine)
    data = Dataset.load_from_df(transactions_data[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = KNNBasic(sim_options={'user_based': True})
    model.fit(trainset)
    return model

# Get the collaborative filtering model on startup
model = get_collaborative_filtering_model()

@app.post("/recommend", response_model=List[RecommendationResponse], tags=["Recommendation"])
def get_recommendations(request: RecommendationRequest, db: Session = Depends(get_db), model: KNNBasic = Depends(get_collaborative_filtering_model)):
    """
    Get personalized product recommendations for a user.
    """
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find similar users
    user_inner_id = model.trainset.to_inner_uid(user.id)
    similar_users = model.get_neighbors(user_inner_id, k=5)  # Adjust 'k' as needed

    # Get products purchased by the user
    user_purchases = transactions_data[transactions_data['user_id'] == user.id]['product_id'].tolist()

    # Get products purchased by similar users
    similar_users_purchases = transactions_data[transactions_data['user_id'].isin(similar_users)]

    # Extract product recommendations with scores
    recommendations = []

    for _, p in similar_users_purchases.iterrows():
        if p.product_id not in user_purchases:
            # Get the predicted rating for the product by the user
            predicted_rating = model.predict(user.id, p.product_id).est
            recommendations.append(RecommendationResponse(product_id=p.product_id, score=predicted_rating))

    return recommendations