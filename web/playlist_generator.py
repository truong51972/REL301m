import json
import numpy as np
from models import SongEmbeddingModel, DQNModel, PlaylistEnvironment
from config import Config
import os

class PlaylistGenerator:
    def __init__(self):
        self.songs_data = []
        self.embeddings = {}
        self.embedding_model = SongEmbeddingModel()
        self.dqn_model = None
        self.environment = None
        
    def load_data(self):
        """Load dữ liệu từ file"""
        if os.path.exists(Config.PLAYLIST_DATA_FILE):
            with open(Config.PLAYLIST_DATA_FILE, 'r', encoding='utf-8') as f:
                all_songs = json.load(f)
            
            # Chỉ lấy 10,000 bài hát mới nhất
            if len(all_songs) > 10000:
                self.songs_data = all_songs[-10000:]  # Lấy 10,000 bài hát cuối
                print(f"Đã load {len(self.songs_data)} bài hát (10,000 bài hát mới nhất từ tổng {len(all_songs)} bài)")
            else:
                self.songs_data = all_songs
                print(f"Đã load {len(self.songs_data)} bài hát")
        else:
            print("Không tìm thấy file dữ liệu. Vui lòng chạy spotify_data_collector.py trước")
            return False
        
        # Load embeddings nếu có
        if os.path.exists(Config.EMBEDDING_FILE):
            self.load_embeddings()
        else:
            print("Không tìm thấy file embeddings. Sẽ tạo mới...")
            self.create_embeddings()
        
        return True
    
    def load_embeddings(self):
        """Load embeddings từ file"""
        self.embeddings = {}
        with open(Config.EMBEDDING_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Bỏ qua dòng header
                parts = line.strip().split()
                if len(parts) > Config.EMBEDDING_DIM:
                    song_id = parts[0]
                    embedding = [float(x) for x in parts[1:Config.EMBEDDING_DIM+1]]
                    self.embeddings[song_id] = embedding
        
        print(f"Đã load {len(self.embeddings)} embeddings")
    
    def load_model(self, model_path):
        """Load model DQN từ file"""
        try:
            from models import DQNModel, PlaylistEnvironment
            
            # Tạo environment nếu chưa có
            if not self.environment:
                self.environment = PlaylistEnvironment(self.songs_data, self.embeddings)
                print("✅ Đã tạo environment")
            
            state_size = Config.EMBEDDING_DIM
            action_size = len(self.songs_data)
            self.dqn_model = DQNModel(state_size, action_size)
            self.dqn_model.load(model_path)
            print(f"✅ Đã load model từ {model_path}")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            return False
    
    def create_embeddings(self):
        """Tạo embeddings cho bài hát"""
        print("Bắt đầu tạo embeddings...")
        
        # Sử dụng dữ liệu cơ bản thay vì audio features
        features = []
        song_ids = []
        
        for song in self.songs_data:
            # Tạo feature vector từ dữ liệu cơ bản
            feature_vector = []
            
            # 1. Popularity (0-100) -> normalize to 0-1
            popularity = song.get('popularity', 50) / 100.0
            feature_vector.append(popularity)
            
            # 2. Duration (normalize to 0-1, max 10 minutes)
            duration = min(song.get('duration_ms', 180000) / 600000.0, 1.0)
            feature_vector.append(duration)
            
            # 3. Explicit flag (0 or 1)
            explicit = float(song.get('explicit', False))
            feature_vector.append(explicit)
            
            # 4. Release year (normalize to 0-1, 1950-2024)
            try:
                year = int(song.get('release_date', '2020')[:4])
                year_normalized = (year - 1950) / (2024 - 1950)
                year_normalized = max(0, min(1, year_normalized))  # Clamp to 0-1
            except:
                year_normalized = 0.5
            feature_vector.append(year_normalized)
            
            # 5. Artist name length (normalize)
            artist_length = len(song.get('artist', '')) / 50.0  # Assume max 50 chars
            feature_vector.append(min(artist_length, 1.0))
            
            # 6. Song name length (normalize)
            name_length = len(song.get('name', '')) / 100.0  # Assume max 100 chars
            feature_vector.append(min(name_length, 1.0))
            
            # 7. Album name length (normalize)
            album_length = len(song.get('album', '')) / 100.0
            feature_vector.append(min(album_length, 1.0))
            
            # 8. Search query category (one-hot encoding)
            search_query = song.get('search_query', '').lower()
            categories = ['vietnamese', 'k-pop', 'j-pop', 'pop', 'rock', 'hip hop', 'electronic']
            for category in categories:
                feature_vector.append(1.0 if category in search_query else 0.0)
            
            # 9. Audio features (nếu có từ dữ liệu cũ)
            audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'tempo', 'instrumentalness', 'speechiness', 'liveness']
            for feature in audio_features:
                value = song.get(feature, 0.5)  # Default to 0.5 if not available
                feature_vector.append(value)
            
            features.append(feature_vector)
            song_ids.append(song['id'])
        
        # Chuyển đổi sang numpy array
        features = np.array(features)
        print(f"Shape của features: {features.shape}")
        
        # Sử dụng PCA để giảm chiều xuống 128
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(128, features.shape[1]))
        embeddings = pca.fit_transform(features)
        
        # Chuyển đổi sang dictionary
        for i, song_id in enumerate(song_ids):
            self.embeddings[song_id] = embeddings[i].tolist()
        
        print(f"Đã tạo {len(self.embeddings)} embeddings với shape {embeddings.shape}")
        
        # Lưu embeddings
        with open('data/embeddings.json', 'w') as f:
            json.dump(self.embeddings, f)
    
    def train_rl_model(self, episodes=1000):
        """Train DQN model"""
        print("Bắt đầu training DQN model...")
        
        # Tạo environment
        self.environment = PlaylistEnvironment(self.songs_data, self.embeddings)
        
        # Tạo DQN model
        state_size = Config.EMBEDDING_DIM
        action_size = len(self.songs_data)
        self.dqn_model = DQNModel(state_size, action_size)
        
        batch_size = 32
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            
            while True:
                # Chọn action
                action = self.dqn_model.act(state, self.environment.available_songs)
                
                # Thực hiện action
                next_state, reward, done, _ = self.environment.step(action)
                
                # Lưu experience
                self.dqn_model.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                # Train model
                if len(self.dqn_model.memory) > batch_size:
                    self.dqn_model.replay(batch_size)
                
                if done:
                    break
            
            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.dqn_model.epsilon:.2f}")
        
        # Lưu model
        self.dqn_model.save('models/dqn_model.h5')
        print("Đã lưu DQN model")
    
    def generate_playlist(self, seed_song_id=None, length=20, constraints=None):
        """Tạo playlist với constraints"""
        if self.environment is None:
            print("Vui lòng train model trước")
            return []
        
        # Reset environment
        state = self.environment.reset()
        
        # Nếu có seed song, thêm vào playlist
        if seed_song_id:
            seed_idx = self.environment.song_id_to_idx.get(seed_song_id)
            if seed_idx is not None and seed_idx in self.environment.available_songs:
                self.environment.current_playlist = [seed_idx]
                self.environment.available_songs.remove(seed_idx)
                state = self.environment._get_state()
        
        # Áp dụng constraints
        if constraints:
            self._apply_constraints(constraints)
        
        # Tạo playlist
        for i in range(length):
            if len(self.environment.available_songs) == 0:
                print(f"Hết bài hát available sau {i} bài")
                break
            
            # Chọn action
            action = self.dqn_model.act(state, self.environment.available_songs)
            print(f"Bài {i+1}: Action={action}, Available songs={len(self.environment.available_songs)}")
            
            # Thực hiện action
            next_state, reward, done, _ = self.environment.step(action)
            state = next_state
            print(f"Reward: {reward}, Done: {done}, Playlist length: {len(self.environment.current_playlist)}")
            
            # Không dừng sớm, chỉ dừng khi hết bài hoặc đủ số lượng
            # if done:
            #     break
        
        return self.environment.get_playlist()
    
    def _apply_constraints(self, constraints):
        """Áp dụng constraints cho playlist (linh hoạt hơn)"""
        available_songs = self.environment.available_songs.copy()
        
        # Đếm số bài hát theo từng constraint để kiểm tra
        genre_counts = {}
        for song_idx in available_songs:
            song = self.songs_data[song_idx]
            genre = song.get('genre', 'Unknown')
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        print(f"Thống kê thể loại có sẵn: {genre_counts}")
        
        for song_idx in available_songs:
            song = self.songs_data[song_idx]
            should_remove = False
            
            # Genre constraint (linh hoạt hơn)
            if 'genre' in constraints and len(constraints['genre']) > 0:
                selected_genres = constraints['genre']
                song_genre = song.get('genre', 'Vietnamese')  # Default Vietnamese
                
                # Nếu chọn nhiều thể loại, chấp nhận bất kỳ thể loại nào trong danh sách
                if song_genre not in selected_genres:
                    # Nếu chọn Vietnamese, chấp nhận tất cả
                    if 'Vietnamese' not in selected_genres:
                        should_remove = True
                    # Nếu chọn Pop/Rock/Hip-hop mà bài hát là Vietnamese, vẫn chấp nhận một số
                    elif song_genre == 'Vietnamese' and len(selected_genres) > 0 and 'Vietnamese' not in selected_genres:
                        # Chỉ loại bỏ 70% bài Vietnamese để vẫn có đủ bài hát
                        import random
                        if random.random() < 0.7:
                            should_remove = True
            
            # Popularity constraint
            if 'min_popularity' in constraints and song.get('popularity', 0) < constraints['min_popularity']:
                should_remove = True
            
            # Year constraint
            if 'min_year' in constraints:
                try:
                    year = int(song.get('release_date', '0')[:4])
                    if year < constraints['min_year']:
                        should_remove = True
                except:
                    should_remove = True
            
            # Audio features constraints
            audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'tempo', 'instrumentalness', 'speechiness', 'liveness']
            
            for feature in audio_features:
                if feature in constraints and song.get(feature) is not None:
                    target_value = constraints[feature]
                    actual_value = song.get(feature, 0)
                    
                    # Tính độ lệch (tolerance ±20 cho các features thường, ±30 cho tempo)
                    tolerance = 30 if feature == 'tempo' else 20
                    
                    if abs(actual_value - target_value) > tolerance:
                        should_remove = True
                        break
            
            if should_remove:
                self.environment.available_songs.remove(song_idx)
        
        # Đảm bảo có ít nhất 50 bài hát để tạo playlist
        if len(self.environment.available_songs) < 50:
            print(f"⚠️ Chỉ còn {len(self.environment.available_songs)} bài hát sau khi áp dụng constraints")
            print("🔄 Thêm lại một số bài hát Vietnamese để đảm bảo đa dạng...")
            
            # Thêm lại một số bài hát Vietnamese
            for song_idx in range(len(self.songs_data)):
                if len(self.environment.available_songs) >= 100:
                    break
                if song_idx not in self.environment.available_songs:
                    song = self.songs_data[song_idx]
                    if song.get('genre', 'Vietnamese') == 'Vietnamese':
                        self.environment.available_songs.append(song_idx)
        
        print(f"✅ Có {len(self.environment.available_songs)} bài hát available sau khi áp dụng constraints")
    
    def evaluate_playlist(self, playlist):
        """Đánh giá chất lượng playlist"""
        if len(playlist) == 0:
            return 0
        
        if len(playlist) == 1:
            # Chỉ có 1 bài hát, đánh giá dựa trên popularity
            popularity = playlist[0].get('popularity', 50)
            return (popularity / 100) * 5  # Tối đa 5 điểm cho 1 bài
        
        # Tính độ tương đồng trung bình
        similarities = []
        for i in range(len(playlist) - 1):
            song1_id = playlist[i]['id']
            song2_id = playlist[i+1]['id']
            
            if song1_id in self.embeddings and song2_id in self.embeddings:
                similarity = np.dot(self.embeddings[song1_id], self.embeddings[song2_id])
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            diversity = 1 - abs(avg_similarity)  # Sử dụng abs để tránh giá trị âm
            popularity = np.mean([song.get('popularity', 50) for song in playlist])
            
            # Score tổng hợp (điều chỉnh để có điểm số hợp lý)
            score = (abs(avg_similarity) * 0.4 + diversity * 0.3 + popularity / 100 * 0.3) * 10
            return max(0, min(10, score))  # Clamp giữa 0-10
        
        # Fallback: dựa trên popularity trung bình
        popularity = np.mean([song.get('popularity', 50) for song in playlist])
        return (popularity / 100) * 7  # Tối đa 7 điểm

def main():
    """Hàm main để test"""
    generator = PlaylistGenerator()
    
    if generator.load_data():
        # Train model
        generator.train_rl_model(episodes=500)
        
        # Tạo playlist mẫu
        constraints = {
            'genre': ['pop', 'vietnamese'],
            'min_popularity': 50,
            'min_year': 2020
        }
        
        playlist = generator.generate_playlist(length=15, constraints=constraints)
        
        print("\nPlaylist được tạo:")
        for i, song in enumerate(playlist, 1):
            print(f"{i}. {song['name']} - {song['artist']} ({song.get('genre', 'Unknown')})")
        
        score = generator.evaluate_playlist(playlist)
        print(f"\nĐiểm đánh giá playlist: {score:.2f}/10")

if __name__ == "__main__":
    main() 