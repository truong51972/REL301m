#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from playlist_generator import PlaylistGenerator
from models import DQNModel, PlaylistEnvironment
from config import Config

def log_message(message, log_file="log_train.txt"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, flush=True)
    
    # Write to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def main():
    # Clear previous log
    with open("log_train.txt", 'w', encoding='utf-8') as f:
        f.write("")
    
    log_message("=== TRAINING DQN MODEL ===")
    log_message("Starting training process...")
    
    # Check data file
    if not os.path.exists(Config.PLAYLIST_DATA_FILE):
        log_message("ERROR: Data file not found!")
        log_message(f"Please run collect_data.py first")
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
        
        # Create environment
        log_message("Creating training environment...")
        generator.environment = PlaylistEnvironment(generator.songs_data, generator.embeddings)
        log_message(f"Environment created with {len(generator.environment.available_songs)} available songs")
        
        # Create DQN model
        state_size = Config.EMBEDDING_DIM
        action_size = len(generator.songs_data)
        generator.dqn_model = DQNModel(state_size, action_size)
        log_message(f"DQN model created: state_size={state_size}, action_size={action_size}")
        
        # Training parameters
        episodes = 30
        log_message(f"Starting training with {episodes} episodes...")
        
        start_time = time.time()
        
        # Training loop
        for episode in range(episodes):
            try:
                # Log progress every 5 episodes
                if episode % 5 == 0:
                    progress = (episode / episodes) * 100
                    log_message(f"Episode {episode}/{episodes} ({progress:.1f}%)")
                
                # Reset environment for new episode
                state = generator.environment.reset()
                total_reward = 0
                steps = 0
                
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
                        steps += 1
                        
                        # Train model if enough experiences
                        if len(generator.dqn_model.memory) > 32:
                            generator.dqn_model.replay(32)
                        
                        # Check if episode is done
                        if done:
                            break
                            
                    except Exception as step_error:
                        log_message(f"Error in step {steps} of episode {episode}: {str(step_error)}")
                        break
                
                # Log episode results
                if episode % 5 == 0:
                    log_message(f"Episode {episode} completed: {steps} steps, reward = {total_reward:.2f}, epsilon = {generator.dqn_model.epsilon:.3f}")
                
            except Exception as episode_error:
                log_message(f"Error in episode {episode}: {str(episode_error)}")
                continue
        
        # Training completed
        end_time = time.time()
        training_time = end_time - start_time
        
        log_message("Training completed successfully!")
        log_message(f"Total training time: {training_time:.2f} seconds")
        
        # Save model
        log_message("Saving model...")
        os.makedirs('models', exist_ok=True)
        generator.dqn_model.save('models/dqn_model.h5')
        log_message("Model saved successfully to models/dqn_model.h5")
        
        # Final statistics
        log_message("=== TRAINING STATISTICS ===")
        log_message(f"Episodes completed: {episodes}")
        log_message(f"Training time: {training_time:.2f} seconds")
        log_message(f"Final epsilon: {generator.dqn_model.epsilon:.3f}")
        log_message(f"Memory buffer size: {len(generator.dqn_model.memory)}")
        log_message("=== TRAINING FINISHED ===")
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}")
        log_message("Training failed!")
        return

if __name__ == "__main__":
    main() 