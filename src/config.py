import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Spotify API Configuration
    SPOTIFY_CLIENT_ID = "3d3e0479cc0543aa9e4185a31a3ad6e9"
    SPOTIFY_CLIENT_SECRET = "33602c4e0e4b4053b1c58b52c7aff578"
    SPOTIFY_REDIRECT_URI = "http://localhost:5000/callback"
    
    # Data Configuration
    DATA_DIR = "data"
    RAW_DATA_FILE = os.path.join(DATA_DIR, "raw_data.txt")
    EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.txt")
    METRICS_FILE = os.path.join(DATA_DIR, "metrics.txt")
    PLAYLIST_DATA_FILE = "spotify_songs.json"
    
    # Model Configuration
    EMBEDDING_DIM = 22  # Sửa lại để khớp với embeddings thực tế
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # RL Configuration
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    # Playlist Configuration
    MAX_PLAYLIST_LENGTH = 50
    MIN_PLAYLIST_LENGTH = 10
    
    # Web Configuration
    FLASK_SECRET_KEY = "your-secret-key-here"
    DEBUG = True
    
    # Data Collection
    TARGET_SONGS_COUNT = 10000
    GENRES = [
        "pop", "rock", "hip-hop", "electronic", "jazz", "classical", 
        "country", "r&b", "folk", "reggae", "blues", "metal"
    ]
    
    # Vietnamese and International music keywords
    VIETNAMESE_KEYWORDS = [
        "vietnamese", "v-pop", "việt nam", "hà nội", "hồ chí minh",
        "sài gòn", "nhạc trẻ", "bolero", "dân ca", "cải lương"
    ]
    
    INTERNATIONAL_KEYWORDS = [
        "k-pop", "j-pop", "mandopop", "thai pop", "indonesian pop",
        "malaysian pop", "filipino pop", "singapore pop"
    ] 