import cv2
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# In a production system, this would be replaced with actual face recognition models
# and databases of known faces. For this demo, we'll use basic face detection.

def detect_face(image):
    """
    Detect a face in the given image and return the face region.
    
    Args:
        image: numpy array representing an image
        
    Returns:
        face_image: cropped face region or None if no face is detected
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the pre-trained face detector model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # If no faces detected, return None
        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            return None
        
        # Process the first detected face (assuming one person per image)
        x, y, w, h = faces[0]
        
        # Extract the face region
        face_image = image[y:y+h, x:x+w]
        
        return face_image
    
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return None

def extract_face_features(face_image):
    """
    Extract features from a face image.
    In a real system, this would use a deep learning model to extract face embeddings.
    For this demo, we'll use a simplified feature extraction.
    
    Args:
        face_image: The face image to extract features from
        
    Returns:
        features: A feature vector representing the face
    """
    try:
        # Resize to a standard size
        face_image = cv2.resize(face_image, (128, 128))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine edge features
        edge_features = np.sqrt(sobelx**2 + sobely**2)
        
        # Apply Local Binary Pattern (LBP) for texture features
        radius = 3
        n_points = 24
        lbp = np.zeros_like(blurred)
        for i in range(radius, blurred.shape[0] - radius):
            for j in range(radius, blurred.shape[1] - radius):
                center = blurred[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + int(radius * np.cos(angle))
                    y = j + int(radius * np.sin(angle))
                    pattern |= (blurred[x, y] > center) << k
                lbp[i, j] = pattern
        
        # Combine all features
        features = np.concatenate([
            edge_features.flatten(),
            lbp.flatten(),
            blurred.flatten()
        ])
        
        # Normalize the features
        features = features / np.linalg.norm(features)
        
        return features
    except Exception as e:
        logger.error(f"Error extracting face features: {str(e)}")
        return None

def verify_face(face_image, stored_face_data=None):
    """
    Verify if the face belongs to an authorized user.
    
    Args:
        face_image: The detected face image to verify
        stored_face_data: The enrolled face data to compare against (optional)
    
    Returns:
        bool: True if verification succeeds, False otherwise
    """
    try:
        # Check if we have a valid face image
        if face_image is None or face_image.size == 0:
            logger.warning("Invalid face image for verification")
            return False
        
        # Check minimum dimensions for a reasonable face
        height, width = face_image.shape[:2]
        if height < 50 or width < 50:
            logger.warning(f"Face too small for verification: {width}x{height}")
            return False
        
        # Extract features from the current face
        current_features = extract_face_features(face_image)
        if current_features is None:
            return False
        
        # If we have stored face data, compare the features
        if stored_face_data is not None:
            try:
                # Convert stored face data to numpy array if it's bytes
                if isinstance(stored_face_data, bytes):
                    stored_face = cv2.imdecode(np.frombuffer(stored_face_data, np.uint8), cv2.IMREAD_COLOR)
                    stored_features = extract_face_features(stored_face)
                else:
                    stored_features = stored_face_data
                
                if stored_features is None:
                    return False
                
                # Calculate similarity between current and stored features
                similarity = cosine_similarity([current_features], [stored_features])[0][0]
                
                # Set a balanced threshold for face matching
                # This threshold is more lenient but still secure
                threshold = 0.75
                
                # Log the similarity score for debugging
                logger.debug(f"Face similarity score: {similarity}")
                
                # Additional checks for very low similarity scores
                if similarity < 0.5:
                    logger.warning(f"Very low similarity score: {similarity}")
                    return False
                
                return similarity >= threshold
                
            except Exception as e:
                logger.error(f"Error comparing face features: {str(e)}")
                return False
        
        # If no stored face data, authentication should fail
        return False
        
    except Exception as e:
        logger.error(f"Error in face verification: {str(e)}")
        return False
