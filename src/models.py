import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import Config
import os

class SongEmbeddingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=Config.EMBEDDING_DIM)
        
    def create_model(self, input_dim):
        """Tạo model để tạo embedding cho bài hát"""
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(Config.EMBEDDING_DIM, activation='tanh')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='mse'
        )
        
        return model
    
    def prepare_features(self, songs_data):
        """Chuẩn bị features cho model"""
        features = []
        
        for song in songs_data:
            if song.get('danceability') is not None:
                feature_vector = [
                    song.get('danceability', 0),
                    song.get('energy', 0),
                    song.get('key', 0),
                    song.get('loudness', 0),
                    song.get('mode', 0),
                    song.get('speechiness', 0),
                    song.get('acousticness', 0),
                    song.get('instrumentalness', 0),
                    song.get('liveness', 0),
                    song.get('valence', 0),
                    song.get('tempo', 0),
                    song.get('popularity', 0) / 100.0,  # Normalize popularity
                    song.get('duration_ms', 0) / 1000000.0  # Convert to minutes
                ]
                features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, songs_data):
        """Train model để tạo embedding"""
        print("Chuẩn bị dữ liệu...")
        features = self.prepare_features(songs_data)
        
        if len(features) == 0:
            print("Không có dữ liệu hợp lệ để train")
            return
        
        print(f"Shape của features: {features.shape}")
        
        # Chuẩn hóa dữ liệu
        features_scaled = self.scaler.fit_transform(features)
        
        # Giảm chiều dữ liệu
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Tạo model
        self.model = self.create_model(features_pca.shape[1])
        
        print("Bắt đầu training...")
        self.model.fit(
            features_pca, features_pca,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            validation_split=0.2,
            verbose=1
        )
        
        # Tạo embeddings
        embeddings = self.model.predict(features_pca)
        
        # Lưu embeddings
        self.save_embeddings(songs_data, embeddings)
        
        return embeddings
    
    def save_embeddings(self, songs_data, embeddings):
        """Lưu embeddings vào file"""
        with open(Config.EMBEDDING_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# embedding: {' '.join([str(x) for x in range(Config.EMBEDDING_DIM)])}\n")
            for i, song in enumerate(songs_data):
                if song.get('danceability') is not None:
                    embedding_str = ' '.join([str(x) for x in embeddings[i]])
                    f.write(f"{song['id']} {embedding_str}\n")
        
        print(f"Đã lưu embeddings vào {Config.EMBEDDING_FILE}")

class DQNModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_END
        self.epsilon_decay = Config.EPSILON_DECAY
        self.learning_rate = Config.LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.loss_history = []  # Lưu loss qua các bước train
    
    def _build_model(self):
        """Xây dựng DQN model"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Cập nhật target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Lưu experience vào memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, available_actions):
        """Chọn action dựa trên epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(available_actions)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        # Chỉ xem xét các action có sẵn
        masked_values = np.full(self.action_size, -np.inf)
        masked_values[available_actions] = act_values[0][available_actions]
        return np.argmax(masked_values)
    
    def replay(self, batch_size):
        """Train model với experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        
        for i, idx in enumerate(minibatch):
            state, action, reward, next_state, done = self.memory[idx]
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        target = self.model.predict(states, verbose=0)
        next_target = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_target[i])
        
        # Lưu loss sau mỗi lần fit
        history = self.model.fit(states, target, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        # Lưu loss ra file để vẽ biểu đồ sau
        with open('loss_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"{loss}\n")
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
    
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)

class PlaylistEnvironment:
    def __init__(self, songs_data, embeddings):
        self.songs_data = songs_data
        self.embeddings = embeddings
        self.song_id_to_idx = {song['id']: i for i, song in enumerate(songs_data)}
        self.idx_to_song_id = {i: song['id'] for i, song in enumerate(songs_data)}
        
        # Không tạo similarity matrix để tiết kiệm RAM
        # Sẽ tính similarity on-demand
        print(f"Environment created with {len(songs_data)} songs")
        
        self.reset()
    
    def _compute_similarity(self, idx1, idx2):
        """Tính similarity giữa 2 bài hát on-demand"""
        # Chuyển index thành song_id
        song_id1 = self.idx_to_song_id.get(idx1)
        song_id2 = self.idx_to_song_id.get(idx2)
        
        if song_id1 in self.embeddings and song_id2 in self.embeddings:
            embedding1 = np.array(self.embeddings[song_id1])
            embedding2 = np.array(self.embeddings[song_id2])
            return np.dot(embedding1, embedding2)
        return 0.0
    
    def reset(self):
        """Reset environment"""
        self.current_playlist = []
        self.available_songs = list(range(len(self.songs_data)))
        self.current_song_idx = np.random.choice(self.available_songs)
        self.current_playlist.append(self.current_song_idx)
        self.available_songs.remove(self.current_song_idx)
        
        return self._get_state()
    
    def _get_state(self):
        """Lấy state hiện tại"""
        if len(self.current_playlist) == 0:
            return np.zeros(Config.EMBEDDING_DIM)
        
        # Tính embedding trung bình của playlist hiện tại
        playlist_embeddings = []
        for idx in self.current_playlist:
            song_id = self.idx_to_song_id.get(idx)
            if song_id in self.embeddings:
                playlist_embeddings.append(self.embeddings[song_id])
        
        if playlist_embeddings:
            return np.mean(playlist_embeddings, axis=0)
        else:
            return np.zeros(Config.EMBEDDING_DIM)
    
    def step(self, action):
        """Thực hiện action và trả về reward"""
        if action >= len(self.songs_data) or action not in self.available_songs:
            return self._get_state(), -10, True, {}  # Penalty cho action không hợp lệ
        
        # Thêm bài hát vào playlist
        self.current_playlist.append(action)
        self.available_songs.remove(action)
        
        # Tính reward dựa trên tính tương đồng
        reward = self._calculate_reward(action)
        
        # Kiểm tra xem có kết thúc không
        done = len(self.current_playlist) >= Config.MAX_PLAYLIST_LENGTH or len(self.available_songs) == 0
        
        return self._get_state(), reward, done, {}
    
    def _calculate_reward(self, new_song_idx):
        """Tính reward cho việc thêm bài hát mới"""
        if len(self.current_playlist) < 2:
            reward = np.random.uniform(0, 2)  # Random reward cho bài hát đầu tiên
            if len(self.current_playlist) == 1:  # Log cho episode đầu tiên
                print(f"      Reward cho bài hát đầu: {reward:.2f}")
            return reward
        
        # Tính độ tương đồng với bài hát cuối cùng trong playlist
        last_song_idx = self.current_playlist[-2]
        similarity = self._compute_similarity(last_song_idx, new_song_idx)
        
        # Base reward dựa trên độ tương đồng
        base_reward = similarity * 5
        
        # Thêm randomness để tránh reward giống nhau
        random_factor = np.random.uniform(0.8, 1.2)
        base_reward *= random_factor
        
        # Thêm reward cho diversity
        diversity_bonus = 0
        if len(self.current_playlist) > 3:
            recent_songs = self.current_playlist[-4:-1]
            similarities = [self._compute_similarity(recent_songs[i], new_song_idx) 
                           for i in range(len(recent_songs))]
            avg_similarity = np.mean(similarities)
            diversity_bonus = (1 - avg_similarity) * 3
            base_reward += diversity_bonus
        
        # Thêm reward cho độ dài playlist
        length_bonus = len(self.current_playlist) * 0.5
        base_reward += length_bonus
        
        # Thêm noise để tạo variation
        noise = np.random.normal(0, 0.5)
        base_reward += noise
        
        # Đảm bảo reward không âm
        final_reward = max(0, base_reward)
        
        # Log chi tiết cho episode đầu tiên
        if len(self.current_playlist) <= 5:  # Chỉ log 5 steps đầu
            print(f"      Step {len(self.current_playlist)}: similarity={similarity:.3f}, base={base_reward:.2f}, diversity={diversity_bonus:.2f}, length={length_bonus:.2f}, noise={noise:.2f}, final={final_reward:.2f}")
        
        return final_reward
    
    def get_playlist(self):
        """Lấy playlist hiện tại"""
        return [self.songs_data[idx] for idx in self.current_playlist] 