#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from playlist_generator import PlaylistGenerator
from models import DQNModel, PlaylistEnvironment
from config import Config

def log_message(message, log_file="log_train_improved.txt"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, flush=True)
    
    # Write to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def calculate_improved_metrics(playlist, generator):
    """Tính toán metrics cải tiến với reward function tối ưu"""
    if len(playlist) < 2:
        return {
            'similarity': 0.0,
            'diversity': 0.0,
            'popularity': 0.0,
            'avg_energy': 0.0,
            'avg_danceability': 0.0,
            'avg_valence': 0.0,
            'tempo_variance': 0.0,
            'genre_diversity': 0.0,
            'total_reward': 0.0
        }
    
    # Lấy features và thông tin của các bài hát trong playlist
    playlist_features = []
    popularities = []
    energies = []
    danceabilities = []
    valences = []
    tempos = []
    
    for song_idx in playlist:
        song = generator.songs_data[song_idx]
        
        # Audio features (chuẩn hóa tốt hơn)
        features = [
            song.get('danceability', 0.5),
            song.get('energy', 0.5),
            song.get('valence', 0.5),
            min(song.get('tempo', 120) / 200.0, 1.0),  # Normalize tempo
            max((song.get('loudness', -10) + 60) / 60.0, 0),  # Normalize loudness
            song.get('speechiness', 0.1),
            song.get('acousticness', 0.5),
            song.get('instrumentalness', 0.1),
            song.get('liveness', 0.1),
            song.get('key', 6) / 11.0,  # Normalize key
            song.get('mode', 0.5),
            (song.get('time_signature', 4) - 3) / 4.0,  # Normalize time signature
            min(song.get('duration_ms', 200000) / 300000.0, 1.0),  # Normalize duration
            song.get('popularity', 50) / 100.0  # Normalize popularity
        ]
        
        # Add more varied features to reach 22
        features.extend([
            song.get('energy', 0.5) * song.get('danceability', 0.5),  # Energy-dance interaction
            song.get('valence', 0.5) * song.get('energy', 0.5),      # Valence-energy interaction
            1.0 - song.get('acousticness', 0.5),                      # Electric-ness
            song.get('tempo', 120) / song.get('duration_ms', 200000) * 1000,  # Tempo density
            song.get('loudness', -10) / song.get('energy', 0.5) if song.get('energy', 0.5) > 0 else 0,  # Loudness per energy
            song.get('speechiness', 0) + song.get('liveness', 0),     # Human factor
            abs(song.get('valence', 0.5) - 0.5) * 2,                 # Emotional intensity
            song.get('instrumentalness', 0) * (1 - song.get('speechiness', 0))  # Pure instrumental
        ])
        
        playlist_features.append(features[:22])  # Ensure exactly 22 features
        popularities.append(song.get('popularity', 50))
        energies.append(song.get('energy', 0.5))
        danceabilities.append(song.get('danceability', 0.5))
        valences.append(song.get('valence', 0.5))
        tempos.append(song.get('tempo', 120))
    
    playlist_features = np.array(playlist_features)
    
    # 1. IMPROVED SIMILARITY SCORE - Tính độ tương đồng với penalty cho quá cao
    similarities = []
    for i in range(len(playlist_features) - 1):
        sim = cosine_similarity(
            playlist_features[i].reshape(1, -1),
            playlist_features[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    # Penalty cho similarity quá cao (>0.9)
    similarity_penalty = max(0, (avg_similarity - 0.9) * 2)
    adjusted_similarity = avg_similarity - similarity_penalty
    
    # 2. ENHANCED DIVERSITY SCORE - Cải thiện tính đa dạng
    feature_variance = np.mean(np.var(playlist_features, axis=0))
    tempo_variance = np.var(tempos) / 1000  # Normalize tempo variance
    energy_variance = np.var(energies)
    valence_variance = np.var(valences)
    
    # Tổng hợp diversity từ nhiều nguồn
    diversity_score = (
        feature_variance * 5 +      # Feature variance
        tempo_variance * 3 +        # Tempo diversity
        energy_variance * 2 +       # Energy diversity
        valence_variance * 2        # Mood diversity
    )
    diversity_score = min(diversity_score, 5.0)  # Cap at 5.0
    
    # 3. BALANCED POPULARITY SCORE - Cân bằng độ phổ biến
    avg_popularity = np.mean(popularities)
    popularity_variance = np.var(popularities)
    
    # Thưởng cho popularity trung bình (50-70) và có variance hợp lý
    if 50 <= avg_popularity <= 70:
        popularity_bonus = 1.2
    elif 40 <= avg_popularity <= 80:
        popularity_bonus = 1.0
    else:
        popularity_bonus = 0.8
    
    popularity_score = (avg_popularity / 100 * 3) * popularity_bonus
    
    # 4. FLOW SCORE - Đánh giá tính liền mạch của playlist
    energy_flow = 1.0 - np.mean(np.abs(np.diff(energies)))  # Smooth energy transition
    tempo_flow = 1.0 - np.mean(np.abs(np.diff(tempos))) / 50  # Smooth tempo transition
    flow_score = (energy_flow + tempo_flow) / 2
    
    # 5. IMPROVED REWARD CALCULATION - Cân bằng tốt hơn
    total_reward = (
        adjusted_similarity * 0.25 * 10 +  # Similarity: 25% (giảm từ 40%)
        diversity_score * 0.35 +           # Diversity: 35% (tăng từ 30%)
        popularity_score * 0.25 +          # Popularity: 25% (giảm từ 30%)
        flow_score * 0.15 * 5              # Flow: 15% (mới thêm)
    )
    
    # Bonus cho playlist cân bằng
    balance_bonus = 0
    if 0.7 <= avg_similarity <= 0.85 and diversity_score >= 1.0:
        balance_bonus = 1.0
    
    total_reward += balance_bonus
    
    return {
        'similarity': avg_similarity,
        'adjusted_similarity': adjusted_similarity,
        'diversity': diversity_score,
        'popularity': avg_popularity,
        'avg_energy': np.mean(energies),
        'avg_danceability': np.mean(danceabilities),
        'avg_valence': np.mean(valences),
        'tempo_variance': tempo_variance,
        'flow_score': flow_score,
        'balance_bonus': balance_bonus,
        'total_reward': total_reward,
        'num_songs': len(playlist)
    }

class ImprovedPlaylistEnvironment(PlaylistEnvironment):
    """Environment cải tiến với reward function tối ưu"""
    
    def __init__(self, songs_data, embeddings):
        super().__init__(songs_data, embeddings)
        self.step_rewards = []
    
    def step(self, action):
        """Override step function với reward cải tiến"""
        if action not in self.available_songs:
            return self.state, -10, True, {}  # Penalty lớn cho invalid action
        
        # Add song to playlist
        self.current_playlist.append(action)
        self.available_songs.remove(action)
        self.step_rewards.append(0)  # Placeholder
        
        # Calculate improved reward
        if len(self.current_playlist) >= 2:
            # Immediate reward cho step này
            last_song = self.songs_data[self.current_playlist[-1]]
            prev_song = self.songs_data[self.current_playlist[-2]]
            
            # Similarity reward (cải tiến)
            last_features = self._get_song_features(last_song)
            prev_features = self._get_song_features(prev_song)
            similarity = cosine_similarity(
                last_features.reshape(1, -1),
                prev_features.reshape(1, -1)
            )[0][0]
            
            # Optimal similarity range: 0.7-0.85
            if 0.7 <= similarity <= 0.85:
                similarity_reward = similarity * 10
            elif similarity > 0.85:
                similarity_reward = (0.85 - (similarity - 0.85) * 2) * 10  # Penalty for too high
            else:
                similarity_reward = similarity * 5  # Lower reward for low similarity
            
            # Diversity reward
            if len(self.current_playlist) >= 3:
                recent_features = [self._get_song_features(self.songs_data[idx]) 
                                 for idx in self.current_playlist[-3:]]
                diversity = np.mean(np.var(recent_features, axis=0)) * 20
                diversity_reward = min(diversity, 5)
            else:
                diversity_reward = 0
            
            # Popularity reward
            popularity = (last_song.get('popularity', 50) + prev_song.get('popularity', 50)) / 2
            if 50 <= popularity <= 70:
                popularity_reward = 3
            elif 40 <= popularity <= 80:
                popularity_reward = 2
            else:
                popularity_reward = 1
            
            # Flow reward (energy/tempo transition)
            energy_diff = abs(last_song.get('energy', 0.5) - prev_song.get('energy', 0.5))
            tempo_diff = abs(last_song.get('tempo', 120) - prev_song.get('tempo', 120)) / 50
            flow_reward = max(0, 3 - energy_diff * 5 - tempo_diff)
            
            step_reward = similarity_reward + diversity_reward + popularity_reward + flow_reward
        else:
            step_reward = 5  # Base reward for first song
        
        self.step_rewards[-1] = step_reward
        
        # Update state
        self.state = self._get_state()
        
        # Check if done
        done = len(self.current_playlist) >= 20 or len(self.available_songs) == 0
        
        return self.state, step_reward, done, {}

def main():
    # Clear previous log
    with open("log_train_improved.txt", 'w', encoding='utf-8') as f:
        f.write("")
    
    log_message("=== IMPROVED TRAINING DQN MODEL ===")
    log_message("Starting improved training with enhanced reward function...")
    log_message("IMPROVEMENTS:")
    log_message("- Reduced similarity weight: 40% -> 25%")
    log_message("- Increased diversity weight: 30% -> 35%")
    log_message("- Added flow score: 15% for smooth transitions")
    log_message("- Penalty for over-similarity (>0.9)")
    log_message("- Enhanced diversity calculation")
    log_message("- Balance bonus for optimal playlists")
    
    # Check data file
    if not os.path.exists(Config.PLAYLIST_DATA_FILE):
        log_message("ERROR: Data file not found!")
        return
    
    log_message("Data file found, proceeding...")
    
    # Initialize generator
    log_message("Loading playlist generator...")
    generator = PlaylistGenerator()
    
    try:
        if not generator.load_data():
            log_message("ERROR: Cannot load data!")
            return
        
        log_message(f"Loaded {len(generator.songs_data)} songs successfully")
        
        # Create embeddings if needed
        if not generator.embeddings:
            log_message("Creating embeddings...")
            generator.create_embeddings()
            log_message("Embeddings created successfully")
        
        # Create IMPROVED environment
        log_message("Creating improved training environment...")
        generator.environment = ImprovedPlaylistEnvironment(generator.songs_data, generator.embeddings)
        log_message(f"Improved environment created with {len(generator.environment.available_songs)} available songs")
        
        # Create DQN model with modified hyperparameters
        state_size = Config.EMBEDDING_DIM
        action_size = len(generator.songs_data)
        generator.dqn_model = DQNModel(state_size, action_size)
        
        # Modify learning parameters for better exploration-exploitation balance
        generator.dqn_model.epsilon = 0.9  # Start with lower epsilon
        generator.dqn_model.epsilon_decay = 0.99  # Slower decay
        generator.dqn_model.learning_rate = 0.0005  # Lower learning rate
        
        log_message(f"Improved DQN model created: state_size={state_size}, action_size={action_size}")
        log_message(f"Modified hyperparameters: epsilon=0.9, decay=0.99, lr=0.0005")
        
        # Training parameters
        episodes = 40  # More episodes for better learning
        log_message(f"Starting training with {episodes} episodes...")
        log_message("=" * 80)
        
        start_time = time.time()
        
        # Training loop
        for episode in range(episodes):
            try:
                episode_start_time = time.time()
                
                # Log progress
                progress = (episode / episodes) * 100
                log_message(f"EPISODE {episode}/{episodes} ({progress:.1f}%) - Starting...")
                
                # Reset environment for new episode
                state = generator.environment.reset()
                total_reward = 0
                steps = 0
                episode_rewards = []
                
                # Episode loop
                while True:
                    try:
                        # Choose action
                        action = generator.dqn_model.act(state, generator.environment.available_songs)
                        
                        # Take step
                        next_state, reward, done, _ = generator.environment.step(action)
                        
                        # Remember experience
                        generator.dqn_model.remember(state, action, reward, next_state, done)
                        
                        # Update state and stats
                        state = next_state
                        total_reward += reward
                        episode_rewards.append(reward)
                        steps += 1
                        
                        # Train model more frequently
                        if len(generator.dqn_model.memory) > 16:
                            generator.dqn_model.replay(16)
                        
                        # Check if episode is done
                        if done or steps >= 20:
                            break
                            
                    except Exception as step_error:
                        log_message(f"  Error in step {steps}: {str(step_error)}")
                        break
                
                # Episode completed - calculate improved metrics
                episode_time = time.time() - episode_start_time
                current_playlist = generator.environment.current_playlist
                
                if len(current_playlist) >= 2:
                    # Calculate improved metrics
                    metrics = calculate_improved_metrics(current_playlist, generator)
                    
                    # Log detailed results
                    log_message(f"EPISODE {episode} COMPLETED:")
                    log_message(f"  Time: {episode_time:.2f}s | Steps: {steps} | Songs: {metrics['num_songs']}")
                    log_message(f"  Total Reward: {total_reward:.3f} | Avg Reward: {total_reward/steps:.3f}")
                    log_message(f"  SIMILARITY: {metrics['similarity']:.4f} -> ADJUSTED: {metrics['adjusted_similarity']:.4f}")
                    log_message(f"  DIVERSITY: {metrics['diversity']:.4f} | FLOW: {metrics['flow_score']:.4f}")
                    log_message(f"  POPULARITY: {metrics['popularity']:.2f} | ENERGY: {metrics['avg_energy']:.4f}")
                    log_message(f"  DANCEABILITY: {metrics['avg_danceability']:.4f} | VALENCE: {metrics['avg_valence']:.4f}")
                    log_message(f"  TEMPO_VAR: {metrics['tempo_variance']:.4f} | BALANCE_BONUS: {metrics['balance_bonus']:.1f}")
                    log_message(f"  Epsilon: {generator.dqn_model.epsilon:.4f} | Memory: {len(generator.dqn_model.memory)}")
                    
                    # Improved reward breakdown
                    similarity_component = metrics['adjusted_similarity'] * 0.25 * 10
                    diversity_component = metrics['diversity'] * 0.35
                    popularity_component = (metrics['popularity'] / 100 * 3) * 0.25
                    flow_component = metrics['flow_score'] * 0.15 * 5
                    
                    log_message(f"  IMPROVED REWARD BREAKDOWN:")
                    log_message(f"    Similarity (25%):  {similarity_component:.3f}")
                    log_message(f"    Diversity (35%):   {diversity_component:.3f}")
                    log_message(f"    Popularity (25%):  {popularity_component:.3f}")
                    log_message(f"    Flow (15%):        {flow_component:.3f}")
                    log_message(f"    Balance Bonus:     {metrics['balance_bonus']:.3f}")
                    log_message(f"    Total Calculated:  {metrics['total_reward']:.3f}")
                    
                else:
                    log_message(f"EPISODE {episode} COMPLETED:")
                    log_message(f"  Time: {episode_time:.2f}s | Steps: {steps} | Songs: {len(current_playlist)}")
                    log_message(f"  Total Reward: {total_reward:.3f} | Insufficient songs for metrics")
                    log_message(f"  Epsilon: {generator.dqn_model.epsilon:.4f} | Memory: {len(generator.dqn_model.memory)}")
                
                log_message("-" * 60)
                
                # Update target network every 10 episodes
                if episode % 10 == 0 and episode > 0:
                    generator.dqn_model.update_target_model()
                    log_message("  Target network updated!")
                
            except Exception as episode_error:
                log_message(f"ERROR in episode {episode}: {str(episode_error)}")
                continue
        
        # Training completed
        end_time = time.time()
        training_time = end_time - start_time
        
        log_message("=" * 80)
        log_message("IMPROVED TRAINING COMPLETED SUCCESSFULLY!")
        log_message(f"Total training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Save improved model
        log_message("Saving improved model...")
        os.makedirs('models', exist_ok=True)
        generator.dqn_model.save('models/dqn_improved_model.h5')
        log_message("Improved model saved to models/dqn_improved_model.h5")
        
        # Final statistics
        log_message("=== IMPROVED TRAINING STATISTICS ===")
        log_message(f"Episodes completed: {episodes}")
        log_message(f"Training time: {training_time:.2f} seconds")
        log_message(f"Average time per episode: {training_time/episodes:.2f} seconds")
        log_message(f"Final epsilon: {generator.dqn_model.epsilon:.4f}")
        log_message(f"Memory buffer size: {len(generator.dqn_model.memory)}")
        log_message(f"Songs in database: {len(generator.songs_data)}")
        log_message("=== IMPROVED TRAINING FINISHED ===")
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}")
        log_message("Training failed!")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main() 