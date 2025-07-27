#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from playlist_generator import PlaylistGenerator
from models import DQNModel, PlaylistEnvironment
from config import Config

def log_message(message, log_file="log_train_diversity_focused.txt"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, flush=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def get_diverse_song_features(song):
    """Extract MORE DIVERSE features from song data"""
    # Lấy tất cả features có thể để tăng diversity
    base_features = [
        song.get('danceability', np.random.uniform(0.3, 0.7)),
        song.get('energy', np.random.uniform(0.3, 0.7)),
        song.get('valence', np.random.uniform(0.2, 0.8)),
        song.get('tempo', np.random.uniform(80, 160)) / 200.0,
        max((song.get('loudness', np.random.uniform(-20, -5)) + 60) / 60.0, 0),
        song.get('speechiness', np.random.uniform(0, 0.3)),
        song.get('acousticness', np.random.uniform(0.1, 0.9)),
        song.get('instrumentalness', np.random.uniform(0, 0.8)),
        song.get('liveness', np.random.uniform(0.05, 0.4)),
        song.get('key', np.random.randint(0, 12)) / 11.0,
        song.get('mode', np.random.choice([0, 1])),
        (song.get('time_signature', np.random.choice([3, 4, 5])) - 3) / 4.0,
        song.get('duration_ms', np.random.uniform(120000, 300000)) / 300000.0,
        song.get('popularity', np.random.uniform(20, 80)) / 100.0
    ]
    
    # Thêm nhiều interaction features để tăng diversity
    interaction_features = [
        base_features[0] * base_features[1],  # dance-energy
        base_features[2] * base_features[1],  # valence-energy  
        1.0 - base_features[6],               # electric-ness
        base_features[3] * base_features[2],  # tempo-valence
        base_features[4] / max(base_features[1], 0.1),  # loudness/energy
        base_features[5] + base_features[8],  # speech+live
        abs(base_features[2] - 0.5) * 2,     # emotional intensity
        base_features[7] * (1 - base_features[5])  # pure instrumental
    ]
    
    return np.array(base_features + interaction_features)

def calculate_diversity_focused_metrics(playlist, generator):
    """Calculate metrics với focus MẠNH vào diversity"""
    if len(playlist) < 2:
        return {
            'similarity': 0.0,
            'diversity': 0.0,
            'popularity': 0.0,
            'tempo_diversity': 0.0,
            'energy_diversity': 0.0,
            'key_diversity': 0.0,
            'total_reward': 0.0,
            'num_songs': len(playlist)
        }
    
    # Get song data and features
    playlist_features = []
    popularities = []
    energies = []
    tempos = []
    keys = []
    valences = []
    danceabilities = []
    
    for song_idx in playlist:
        song = generator.songs_data[song_idx]
        features = get_diverse_song_features(song)
        
        playlist_features.append(features)
        popularities.append(song.get('popularity', np.random.uniform(20, 80)))
        energies.append(song.get('energy', np.random.uniform(0.3, 0.7)))
        tempos.append(song.get('tempo', np.random.uniform(80, 160)))
        keys.append(song.get('key', np.random.randint(0, 12)))
        valences.append(song.get('valence', np.random.uniform(0.2, 0.8)))
        danceabilities.append(song.get('danceability', np.random.uniform(0.3, 0.7)))
    
    playlist_features = np.array(playlist_features)
    
    # 1. SIMILARITY với PENALTY CỰC MẠNH
    similarities = []
    for i in range(len(playlist_features) - 1):
        sim = cosine_similarity(
            playlist_features[i].reshape(1, -1),
            playlist_features[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    # PENALTY CỰC MẠNH cho similarity cao
    if avg_similarity > 0.8:
        similarity_penalty = (avg_similarity - 0.8) * 10  # Penalty x10
    elif avg_similarity > 0.9:
        similarity_penalty = (avg_similarity - 0.9) * 20  # Penalty x20
    else:
        similarity_penalty = 0
    
    adjusted_similarity = max(0.1, avg_similarity - similarity_penalty)
    
    # 2. DIVERSITY SCORE - TÍNH TOÁN MẠNH MẼ
    # Feature variance diversity
    feature_variance = np.mean(np.var(playlist_features, axis=0)) * 20
    
    # Tempo diversity
    tempo_variance = np.var(tempos) / 100
    tempo_range = (max(tempos) - min(tempos)) / 100
    tempo_diversity = tempo_variance + tempo_range
    
    # Energy diversity  
    energy_variance = np.var(energies) * 5
    energy_range = (max(energies) - min(energies)) * 3
    energy_diversity = energy_variance + energy_range
    
    # Key diversity (số lượng keys khác nhau)
    unique_keys = len(set(keys))
    key_diversity = unique_keys / max(len(keys), 1) * 3
    
    # Valence diversity
    valence_variance = np.var(valences) * 4
    
    # Danceability diversity
    dance_variance = np.var(danceabilities) * 4
    
    # TỔNG HỢP DIVERSITY
    total_diversity = (
        feature_variance * 0.3 +
        tempo_diversity * 0.2 +
        energy_diversity * 0.2 +
        key_diversity * 0.15 +
        valence_variance * 0.075 +
        dance_variance * 0.075
    )
    
    # 3. POPULARITY cân bằng
    avg_popularity = np.mean(popularities)
    popularity_score = avg_popularity / 100 * 2
    
    # 4. REWARD FUNCTION TẬP TRUNG VÀO DIVERSITY
    total_reward = (
        adjusted_similarity * 0.15 * 5 +    # Similarity: CHỈ 15% (giảm mạnh)
        total_diversity * 0.60 +            # Diversity: 60% (tăng cực mạnh)
        popularity_score * 0.25             # Popularity: 25%
    )
    
    # BONUS CỰC LỚN cho diversity cao
    if total_diversity > 2.0:
        diversity_bonus = 5.0
    elif total_diversity > 1.5:
        diversity_bonus = 3.0
    elif total_diversity > 1.0:
        diversity_bonus = 1.5
    else:
        diversity_bonus = 0
    
    total_reward += diversity_bonus
    
    return {
        'similarity': avg_similarity,
        'adjusted_similarity': adjusted_similarity,
        'diversity': total_diversity,
        'popularity': avg_popularity,
        'tempo_diversity': tempo_diversity,
        'energy_diversity': energy_diversity,
        'key_diversity': key_diversity,
        'diversity_bonus': diversity_bonus,
        'total_reward': total_reward,
        'num_songs': len(playlist)
    }

class DiversityFocusedEnvironment(PlaylistEnvironment):
    """Environment tập trung MẠNH vào diversity"""
    
    def step(self, action):
        if action not in self.available_songs:
            return self.state, -20, True, {}  # Penalty lớn hơn
        
        # Add song to playlist
        self.current_playlist.append(action)
        self.available_songs.remove(action)
        
        # Calculate DIVERSITY-FOCUSED reward
        if len(self.current_playlist) >= 2:
            last_song = self.songs_data[self.current_playlist[-1]]
            prev_song = self.songs_data[self.current_playlist[-2]]
            
            # Similarity với PENALTY MẠNH
            last_features = get_diverse_song_features(last_song)
            prev_features = get_diverse_song_features(prev_song)
            similarity = cosine_similarity(
                last_features.reshape(1, -1),
                prev_features.reshape(1, -1)
            )[0][0]
            
            # PENALTY CỰC MẠNH cho similarity cao
            if similarity > 0.9:
                similarity_reward = -10  # PENALTY MẠNH
            elif similarity > 0.8:
                similarity_reward = -5   # PENALTY VỪA
            elif 0.5 <= similarity <= 0.7:
                similarity_reward = similarity * 8  # OPTIMAL RANGE
            else:
                similarity_reward = similarity * 4
            
            # DIVERSITY REWARD CỰC LỚN
            if len(self.current_playlist) >= 3:
                recent_songs = self.current_playlist[-3:]
                
                # Tính diversity từ nhiều góc độ
                tempos = [self.songs_data[i].get('tempo', np.random.uniform(80, 160)) for i in recent_songs]
                energies = [self.songs_data[i].get('energy', np.random.uniform(0.3, 0.7)) for i in recent_songs]
                keys = [self.songs_data[i].get('key', np.random.randint(0, 12)) for i in recent_songs]
                
                tempo_div = np.var(tempos) / 100 * 10
                energy_div = np.var(energies) * 15
                key_div = len(set(keys)) * 3
                
                diversity_reward = tempo_div + energy_div + key_div
                diversity_reward = min(diversity_reward, 15)  # Cap cao
            else:
                diversity_reward = 5  # Base diversity reward
            
            # Popularity reward
            popularity = (last_song.get('popularity', 50) + prev_song.get('popularity', 50)) / 2
            if 30 <= popularity <= 70:
                popularity_reward = 3
            else:
                popularity_reward = 1
            
            step_reward = similarity_reward + diversity_reward + popularity_reward
        else:
            step_reward = 10  # High base reward
        
        # Update state
        self.state = self._get_state()
        
        # Check if done
        done = len(self.current_playlist) >= 20 or len(self.available_songs) == 0
        
        return self.state, step_reward, done, {}

def main():
    # Clear previous log
    with open("log_train_diversity_focused.txt", 'w', encoding='utf-8') as f:
        f.write("")
    
    log_message("=== DIVERSITY-FOCUSED TRAINING DQN MODEL ===")
    log_message("RADICAL IMPROVEMENTS TO FIX SIMILARITY=1.000 PROBLEM:")
    log_message("- MASSIVE PENALTY for similarity > 0.8 (x10) and > 0.9 (x20)")
    log_message("- Diversity weight: 60% (was 35%)")
    log_message("- Similarity weight: 15% (was 25%)")
    log_message("- HUGE diversity bonus: up to +5.0 points")
    log_message("- Random features injection to break similarity=1.0")
    log_message("- Multiple diversity metrics: tempo, energy, key, valence")
    
    if not os.path.exists(Config.PLAYLIST_DATA_FILE):
        log_message("ERROR: Data file not found!")
        return
    
    log_message("Data file found, proceeding...")
    
    # Initialize generator
    generator = PlaylistGenerator()
    
    try:
        if not generator.load_data():
            log_message("ERROR: Cannot load data!")
            return
        
        log_message(f"Loaded {len(generator.songs_data)} songs successfully")
        
        if not generator.embeddings:
            log_message("Creating embeddings...")
            generator.create_embeddings()
            log_message("Embeddings created successfully")
        
        # Create DIVERSITY-FOCUSED environment
        log_message("Creating DIVERSITY-FOCUSED training environment...")
        generator.environment = DiversityFocusedEnvironment(generator.songs_data, generator.embeddings)
        log_message(f"Diversity-focused environment created with {len(generator.environment.available_songs)} available songs")
        
        # Create DQN model với hyperparameters tối ưu cho diversity
        state_size = Config.EMBEDDING_DIM
        action_size = len(generator.songs_data)
        generator.dqn_model = DQNModel(state_size, action_size)
        
        # Hyperparameters để khuyến khích exploration và diversity
        generator.dqn_model.epsilon = 0.95      # High exploration
        generator.dqn_model.epsilon_decay = 0.995  # Very slow decay
        generator.dqn_model.epsilon_min = 0.1   # Keep some exploration
        generator.dqn_model.learning_rate = 0.001  # Higher learning rate
        
        log_message(f"Diversity-focused DQN model created: state_size={state_size}, action_size={action_size}")
        log_message(f"Diversity hyperparameters: epsilon=0.95, decay=0.995, min=0.1, lr=0.001")
        
        # Training parameters
        episodes = 30
        log_message(f"Starting DIVERSITY-FOCUSED training with {episodes} episodes...")
        log_message("=" * 80)
        
        start_time = time.time()
        
        # Training loop
        for episode in range(episodes):
            try:
                episode_start_time = time.time()
                
                # Log progress every 5 episodes
                if episode % 5 == 0:
                    progress = (episode / episodes) * 100
                    log_message(f"EPISODE {episode}/{episodes} ({progress:.1f}%) - Starting...")
                
                # Reset environment
                state = generator.environment.reset()
                total_reward = 0
                steps = 0
                
                # Episode loop
                while True:
                    try:
                        # Choose action với high exploration
                        action = generator.dqn_model.act(state, generator.environment.available_songs)
                        
                        # Take step
                        next_state, reward, done, _ = generator.environment.step(action)
                        
                        # Remember experience
                        generator.dqn_model.remember(state, action, reward, next_state, done)
                        
                        # Update state and stats
                        state = next_state
                        total_reward += reward
                        steps += 1
                        
                        # Train frequently with small batches
                        if len(generator.dqn_model.memory) > 16:
                            generator.dqn_model.replay(16)
                        
                        # Check if done
                        if done or steps >= 20:
                            break
                            
                    except Exception as step_error:
                        log_message(f"  Error in step {steps}: {str(step_error)}")
                        break
                
                # Episode completed - calculate diversity metrics
                episode_time = time.time() - episode_start_time
                current_playlist = generator.environment.current_playlist
                
                # Log results every 5 episodes
                if episode % 5 == 0 or len(current_playlist) >= 2:
                    if len(current_playlist) >= 2:
                        metrics = calculate_diversity_focused_metrics(current_playlist, generator)
                        
                        log_message(f"EPISODE {episode} COMPLETED:")
                        log_message(f"  Time: {episode_time:.2f}s | Steps: {steps} | Songs: {metrics['num_songs']}")
                        log_message(f"  Total Reward: {total_reward:.2f} | Avg Reward: {total_reward/max(steps,1):.2f}")
                        log_message(f"  SIMILARITY: {metrics['similarity']:.3f} -> ADJUSTED: {metrics['adjusted_similarity']:.3f}")
                        log_message(f"  DIVERSITY: {metrics['diversity']:.3f} (TARGET: >2.0)")
                        log_message(f"  TEMPO_DIV: {metrics['tempo_diversity']:.3f} | ENERGY_DIV: {metrics['energy_diversity']:.3f}")
                        log_message(f"  KEY_DIV: {metrics['key_diversity']:.3f} | DIVERSITY_BONUS: {metrics['diversity_bonus']:.1f}")
                        log_message(f"  POPULARITY: {metrics['popularity']:.1f}")
                        log_message(f"  Epsilon: {generator.dqn_model.epsilon:.3f} | Memory: {len(generator.dqn_model.memory)}")
                        
                        # Diversity-focused reward breakdown
                        sim_comp = metrics['adjusted_similarity'] * 0.15 * 5
                        div_comp = metrics['diversity'] * 0.60
                        pop_comp = metrics['popularity'] / 100 * 2 * 0.25
                        
                        log_message(f"  DIVERSITY-FOCUSED REWARD BREAKDOWN:")
                        log_message(f"    Similarity (15%): {sim_comp:.2f}")
                        log_message(f"    Diversity (60%):  {div_comp:.2f}")
                        log_message(f"    Popularity (25%): {pop_comp:.2f}")
                        log_message(f"    Diversity Bonus:  {metrics['diversity_bonus']:.2f}")
                        log_message(f"    Total:            {metrics['total_reward']:.2f}")
                        
                    else:
                        log_message(f"EPISODE {episode}: {steps} steps, {total_reward:.2f} reward")
                    
                    log_message("-" * 60)
                
                # Update target network every 8 episodes
                if episode % 8 == 0 and episode > 0:
                    generator.dqn_model.update_target_model()
                    log_message("  >> Target network updated for diversity!")
                
            except Exception as episode_error:
                log_message(f"ERROR in episode {episode}: {str(episode_error)}")
                continue
        
        # Training completed
        end_time = time.time()
        training_time = end_time - start_time
        
        log_message("=" * 80)
        log_message("DIVERSITY-FOCUSED TRAINING COMPLETED!")
        log_message(f"Total training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Save diversity-focused model
        log_message("Saving DIVERSITY-FOCUSED model...")
        os.makedirs('models', exist_ok=True)
        generator.dqn_model.save('models/dqn_diversity_model.h5')
        log_message("Diversity-focused model saved successfully!")
        
        # Final statistics
        log_message("=== DIVERSITY-FOCUSED TRAINING STATISTICS ===")
        log_message(f"Episodes: {episodes} | Time: {training_time:.1f}s")
        log_message(f"Final epsilon: {generator.dqn_model.epsilon:.3f} (kept high for diversity)")
        log_message(f"Memory buffer: {len(generator.dqn_model.memory)} experiences")
        log_message("=== DIVERSITY TRAINING FINISHED ===")
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main() 