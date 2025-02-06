from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import logging
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import DictCursor
import os
from typing import Tuple, Optional, List
from functools import wraps
from dotenv import load_dotenv
import requests
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'face_recognition_db')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
    FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD', '0.4'))
    API_USERNAME = os.getenv('API_USERNAME', 'admin')
    API_PASSWORD = os.getenv('API_PASSWORD', 'password')

@dataclass
class MatchRecord:
    created_at: str
    match_score: float
    embedding: List[float]
    id: int
    face_unique_id: int

@dataclass
class MatchResult:
    embedding: List[float]
    match_score: float
    recognition_status: bool
    matched_count: int
    matched_records: List[MatchRecord]

class DatabaseError(Exception):
    """Database-related errors"""
    pass

class FaceProcessingError(Exception):
    """Face processing errors"""
    pass

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            dbname=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise DatabaseError(f"Failed to connect to database: {str(e)}")

def process_image(image_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Process image and extract face encoding"""
    try:
        # Convert image bytes to numpy array
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) != 1:
            raise FaceProcessingError(
                f"Expected 1 face, found {len(face_locations)} faces"
            )
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        if not face_encodings:
            raise FaceProcessingError("Failed to generate face encoding")
            
        return frame, face_encodings[0]
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise FaceProcessingError(f"Failed to process image: {str(e)}")

def find_matching_face(face_encoding: np.ndarray) -> MatchResult:
    """Find matching faces in database from last 24 hours"""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Get embeddings from last 24 hours with creation time
            cur.execute("""
                SELECT id, embedding_vector, created_at, face_unique_id
                FROM hks_face_embedding 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
            """)
            
            rows = cur.fetchall()
            if not rows:
                logger.info("No embeddings found in last 24 hours")
                return MatchResult(
                    embedding=face_encoding.tolist(),
                    match_score=0.0,
                    recognition_status=False,
                    matched_count=0,
                    matched_records=[]
                )
            
            # Process stored embeddings
            stored_embeddings = []
            record_details = []
            for row in rows:
                embedding_array = np.array([float(str(x)) for x in row['embedding_vector']], dtype=np.float64)
                stored_embeddings.append(embedding_array)
                record_details.append({
                        'created_at': row['created_at'],
                        'embedding': embedding_array,
                        'id': row['id'],  # Added
                        'face_unique_id': row['face_unique_id']  # Added
                    })

            
            # Convert input encoding to float64
            face_encoding = face_encoding.astype(np.float64)
            
            # Calculate face distances
            distances = face_recognition.face_distance(stored_embeddings, face_encoding)
            
            # Find matches above threshold
            matches = []
            threshold = 1 - Config.FACE_MATCH_THRESHOLD
            
            for i, distance in enumerate(distances):
                match_score = 1 - distance
                if match_score >= threshold:
                    matches.append(MatchRecord(
                        created_at=record_details[i]['created_at'].isoformat(),
                        match_score=float(match_score),
                        embedding=record_details[i]['embedding'].tolist(),
                        id=record_details[i]['id'],  # Added
                        face_unique_id=record_details[i]['face_unique_id']  # Added
                    ))
            
            if matches:
                # Sort matches by score descending
                matches.sort(key=lambda x: x.match_score, reverse=True)
                best_match = matches[0]
                
                return MatchResult(
                    embedding=face_encoding.tolist(),
                    match_score=best_match.match_score,
                    recognition_status=True,
                    matched_count=len(matches),
                    matched_records=matches
                )
            
            return MatchResult(
                embedding=face_encoding.tolist(),
                match_score=0.0,
                recognition_status=False,
                matched_count=0,
                matched_records=[]
            )
                
    except Exception as e:
        logger.error(f"Error finding matching face: {str(e)}")
        raise FaceProcessingError(f"Failed to find matching face: {str(e)}")
    finally:
        if conn:
            conn.close()

def get_image_from_url(url: str) -> bytes:
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Error downloading image from URL: {str(e)}")
        raise ValueError(f"Failed to download image from URL: {str(e)}")

@app.route("/api/match-face", methods=["POST"])
def match_face():
    """API endpoint to match face against database"""
    try:
        if request.is_json and 'image_url' in request.json:
            image_url = request.json['image_url']
            try:
                image_data = get_image_from_url(image_url)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
        else:
            return jsonify({
                'error': 'No image URL provided'
            }), 400
            
        # Process image and get face encoding
        _, face_encoding = process_image(image_data)
        
        # Find matching faces
        result = find_matching_face(face_encoding)
        
        return jsonify({
            'embedding': result.embedding,
            'match_score': result.match_score,
            'recognition_status': result.recognition_status,
            'matched_count': result.matched_count,
            'matched_records': [
                {
                    'created_at': record.created_at,
                    'match_score': record.match_score,
                    'embedding': record.embedding,
                    'id': record.id,  # Added
                    'face_unique_id': record.face_unique_id  # Added
                } for record in result.matched_records
            ]
        })
    
    except FaceProcessingError as e:
        logger.error(f"Face processing error: {str(e)}")
        return jsonify({'error': str(e)}), 400
        
    except DatabaseError as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3030)
    