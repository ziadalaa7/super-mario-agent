"""
Play Super Mario Bros using a trained RL agent.
"""

import os
from wrappers import make_mario_env, get_action_name
from agent import MarioAgent


def play_game(model_path='./train/best_model_1000000', episodes=1, max_steps=None):
    """
    Play Super Mario Bros using a trained agent.
    
    Args:
        model_path (str): Path to the trained model
        episodes (int): Number of episodes to play
        max_steps (int): Maximum steps per episode (None for infinite)
    """
    # Check if model exists
    if not os.path.exists(model_path + '.zip'):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Create environment
    print("Creating environment...")
    env = make_mario_env()
    
    # Load the agent
    print(f"Loading model from {model_path}...")
    agent = MarioAgent.load(model_path, env)
    
    print(f"Playing {episodes} episode(s)...")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done:
            # Predict action
            action, _ = agent.predict(state, deterministic=True)
            
            # Take step
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Render the game
            try:
                env.venv.envs[0].env.unwrapped.render()
            except:
                pass
            
            # Check max steps
            if max_steps is not None and step >= max_steps:
                done = True
            
            # Print progress periodically
            if step % 100 == 0:
                print(f"Step: {step}, Reward: {episode_reward}")
        
        print(f"Episode {episode + 1} finished!")
        print(f"Total steps: {step}, Total reward: {episode_reward}")
    
    env.close()
    print("\nGame finished!")


if __name__ == '__main__':
    # Play with the best trained model
    play_game(
        model_path='./train/best_model_1000000',
        episodes=1,
        max_steps=None
    )
