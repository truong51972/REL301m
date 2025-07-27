import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json
import time
import random
from tqdm import tqdm
from config import Config
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyDataCollector:
    def __init__(self):
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=Config.SPOTIFY_CLIENT_ID,
                client_secret=Config.SPOTIFY_CLIENT_SECRET
            )
        )
        self.songs_data = []
        
    def search_songs(self, query, limit=50, offset=0):
        """Tìm kiếm bài hát với query"""
        try:
            results = self.sp.search(q=query, type='track', limit=limit, offset=offset)
            return results['tracks']['items']
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm '{query}': {e}")
            return []
    
    def _extract_genre_from_query(self, query):
        """Extract genre từ search query"""
        query_lower = query.lower()
        
        # Vietnamese music
        if any(keyword in query_lower for keyword in ['vietnamese', 'v-pop', 'nhạc', 'bolero', 'cải lương', 'nhạc vàng', 'nhạc sến']):
            return 'Vietnamese'
        
        # K-pop
        if any(keyword in query_lower for keyword in ['k-pop', 'bts', 'blackpink', 'twice', 'red velvet', 'exo', 'bigbang', '2ne1', 'wonder girls', 'girls generation', 'newjeans', 'ive', 'le sserafim', 'aespa', 'itzy', 'stray kids', 'seventeen', 'nct', 'ateez', 'enhypen']):
            return 'K-pop'
        
        # J-pop
        if 'j-pop' in query_lower:
            return 'J-pop'
        
        # Pop
        if 'pop' in query_lower:
            return 'Pop'
        
        # Rock
        if 'rock' in query_lower:
            return 'Rock'
        
        # Hip Hop
        if any(keyword in query_lower for keyword in ['hip hop', 'rap', 'drake', 'post malone', 'travis scott', 'kendrick lamar', 'j cole', 'eminem', 'snoop dogg', 'dr dre', '50 cent', 'jay z', 'kanye west', 'lil wayne', 'future', 'youngboy never broke again']):
            return 'Hip Hop'
        
        # Electronic
        if any(keyword in query_lower for keyword in ['electronic', 'house', 'techno', 'trance', 'dubstep', 'drum and bass', 'ambient', 'chillout', 'david guetta', 'skrillex', 'marshmello', 'alan walker', 'kygo', 'avicii', 'zedd', 'swedish house mafia', 'deadmau5', 'tiesto', 'afrojack']):
            return 'Electronic'
        
        # R&B
        if 'r&b' in query_lower:
            return 'R&B'
        
        # Country
        if 'country' in query_lower:
            return 'Country'
        
        # Jazz
        if 'jazz' in query_lower:
            return 'Jazz'
        
        # Classical
        if 'classical' in query_lower:
            return 'Classical'
        
        # Folk
        if 'folk' in query_lower:
            return 'Folk'
        
        # Reggae
        if 'reggae' in query_lower:
            return 'Reggae'
        
        # Blues
        if 'blues' in query_lower:
            return 'Blues'
        
        # Metal
        if 'metal' in query_lower:
            return 'Metal'
        
        # Default
        return 'Pop'
    
    def get_audio_features(self, track_ids):
        """Lấy audio features cho danh sách track IDs - DEPRECATED"""
        logger.warning("Audio features API đã bị Spotify deprecated từ 27/11/2024")
        logger.info("Sử dụng dữ liệu cơ bản thay thế")
        return [None] * len(track_ids)
    
    def collect_data(self, target_count=2000, progress_callback=None):
        """Thu thập dữ liệu bài hát"""
        logger.info(f"Bắt đầu thu thập {target_count} bài hát...")
        
        # Danh sách từ khóa tìm kiếm (mở rộng để đạt 2k bài hát)
        search_queries = [
            # Vietnamese music
            "vietnamese pop", "v-pop", "nhạc trẻ việt nam", "bolero việt nam",
            "dân ca việt nam", "cải lương", "nhạc vàng", "nhạc sến",
            "sơn tùng", "minh hằng", "đàm vĩnh hưng", "lam trường",
            "hồ ngọc hà", "quang hùng", "đan trường", "hồ quỳnh hương",
            "mỹ tâm", "đàm vĩnh hưng", "quang hùng", "đan trường",
            "hồ quỳnh hương", "mỹ tâm", "đàm vĩnh hưng", "quang hùng",
            "đan trường", "hồ quỳnh hương", "mỹ tâm", "đàm vĩnh hưng",
            
            # International music
            "k-pop", "j-pop", "mandopop", "thai pop", "indonesian pop",
            "malaysian pop", "filipino pop", "singapore pop",
            "bts", "blackpink", "twice", "red velvet", "exo",
            "bigbang", "2ne1", "wonder girls", "girls generation",
            "newjeans", "ive", "le sserafim", "aespa", "itzy",
            "stray kids", "seventeen", "nct", "ateez", "enhypen",
            
            # Popular genres
            "pop hits", "rock hits", "hip hop hits", "electronic hits",
            "jazz hits", "classical hits", "country hits", "r&b hits",
            "folk hits", "reggae hits", "blues hits", "metal hits",
            "indie pop", "alternative rock", "punk rock", "hard rock",
            "soft rock", "progressive rock", "psychedelic rock",
            "funk", "soul", "disco", "house", "techno", "trance",
            "dubstep", "drum and bass", "ambient", "chillout",
            
            # Specific artists
            "taylor swift", "ed sheeran", "justin bieber", "ariana grande",
            "beyonce", "rihanna", "lady gaga", "katy perry",
            "bruno mars", "the weeknd", "drake", "post malone",
            "billie eilish", "dua lipa", "olivia rodrigo", "doja cat",
            "harry styles", "lizzo", "megan thee stallion", "cardi b",
            "nicki minaj", "travis scott", "kendrick lamar", "j cole",
            "eminem", "snoop dogg", "dr dre", "50 cent", "jay z",
            "kanye west", "lil wayne", "future", "youngboy never broke again",
            
            # More artists
            "coldplay", "maroon 5", "imagine dragons", "one republic",
            "the chainsmokers", "calvin harris", "david guetta", "skrillex",
            "marshmello", "alan walker", "kygo", "avicii", "zedd",
            "swedish house mafia", "deadmau5", "tiesto", "afrojack",
            
            # Decades
            "80s hits", "90s hits", "2000s hits", "2010s hits", "2020s hits",
            "classic rock", "oldies", "golden oldies", "vintage pop",
            
            # Moods
            "happy songs", "sad songs", "romantic songs", "party songs",
            "workout songs", "chill songs", "energetic songs", "relaxing songs",
            "motivational songs", "uplifting songs", "feel good songs",
            
            # Languages
            "spanish pop", "french pop", "german pop", "italian pop",
            "portuguese pop", "russian pop", "korean pop", "japanese pop",
            "chinese pop", "thai pop", "indonesian pop", "malaysian pop",
            "filipino pop", "singapore pop", "vietnamese pop"
        ]
        
        collected_count = 0
        seen_tracks = set()
        
        with tqdm(total=target_count, desc="Thu thập bài hát") as pbar:
            # Lặp qua danh sách queries nhiều lần nếu cần
            query_index = 0
            offset = 0
            while collected_count < target_count:
                query = search_queries[query_index % len(search_queries)]
                
                logger.info(f"Tìm kiếm: {query} (offset: {offset})")
                tracks = self.search_songs(query, limit=50, offset=offset)  # Spotify API chỉ cho phép tối đa 50
                
                # Nếu không có bài hát nào, chuyển sang query tiếp theo
                if not tracks:
                    query_index += 1
                    offset = 0
                    continue
                
                for track in tracks:
                    if collected_count >= target_count:
                        break
                        
                    track_id = track['id']
                    if track_id in seen_tracks:
                        continue
                    
                    seen_tracks.add(track_id)
                    
                    # Lấy thông tin cơ bản
                    song_data = {
                        'id': track_id,
                        'name': track['name'],
                        'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                        'album': track['album']['name'] if track['album'] else 'Unknown',
                        'popularity': track['popularity'],
                        'duration_ms': track['duration_ms'],
                        'explicit': track['explicit'],
                        'release_date': track['album']['release_date'] if track['album'] else None,
                        'search_query': query,
                        'genre': self._extract_genre_from_query(query)
                    }
                    
                    # Thêm vào danh sách
                    self.songs_data.append(song_data)
                    collected_count += 1
                    pbar.update(1)
                    
                    # Cập nhật progress callback nếu có
                    if progress_callback:
                        progress_callback(collected_count, target_count, f"Đã thu thập: {song_data['name']} - {song_data['artist']}")
                
                # Tăng offset để lấy thêm bài hát từ cùng query
                offset += 50
                
                # Nếu offset quá lớn hoặc không có bài hát nào, chuyển sang query tiếp theo
                if len(tracks) == 0 or offset >= 1000:  # Spotify có giới hạn khoảng 1000 kết quả
                    query_index += 1
                    offset = 0
                    
                    # Nếu đã lặp qua tất cả queries mà vẫn chưa đủ, dừng lại
                    if query_index >= len(search_queries) * 2:  # Lặp tối đa 2 lần qua tất cả queries
                        logger.warning(f"Đã lặp qua tất cả queries 2 lần, chỉ thu thập được {collected_count} bài hát")
                        break
                
                # Rate limiting
                time.sleep(0.5)
        
        logger.info(f"Đã thu thập {len(self.songs_data)} bài hát")
        return self.songs_data
    
    def save_data(self, filename='spotify_songs.json'):
        """Lưu dữ liệu vào file JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.songs_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu {len(self.songs_data)} bài hát vào {filename}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu file: {e}")
    
    def load_data(self, filename='spotify_songs.json'):
        """Tải dữ liệu từ file JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.songs_data = json.load(f)
            logger.info(f"Đã tải {len(self.songs_data)} bài hát từ {filename}")
            return self.songs_data
        except Exception as e:
            logger.error(f"Lỗi khi tải file: {e}")
            return []

# Test function
if __name__ == "__main__":
    collector = SpotifyDataCollector()
    songs = collector.collect_data(target_count=2000)
    collector.save_data()
    print(f"Đã thu thập {len(songs)} bài hát") 