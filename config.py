import os

# Database configuration
# We support both PostgreSQL and MongoDB
# To switch between them, just change this flag
USE_MONGODB = os.environ.get('USE_MONGODB', 'True').lower() in ('true', '1', 't')

# PostgreSQL database URL from environment variables with connection pooling
POSTGRES_URL = os.environ.get("DATABASE_URL", "")
if POSTGRES_URL:
    # Replace the standard URL with the pooled URL
    POSTGRES_URL = POSTGRES_URL.replace('.us-west-2', '-pooler.us-west-2')

# MongoDB configuration
MONGODB_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGODB_PORT = int(os.environ.get('MONGODB_PORT', 27017))
MONGODB_DB = os.environ.get('MONGODB_DB', 'visionid')
MONGODB_URI = os.environ.get('MONGODB_URI', f'mongodb://{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_DB}')

# Secret key for sessions
SECRET_KEY = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Application configuration
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))

# File storage paths
FACE_IMAGE_DIR = os.path.join('static', 'faces')