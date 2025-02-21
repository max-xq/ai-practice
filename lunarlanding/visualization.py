import torch
import imageio
import gymnasium as gym
from agent import Agent

# Load the trained agent
state_size = 8  # LunarLander state size
action_size = 4  # LunarLander action size
agent = Agent(state_size, action_size)  # Instantiate the agent
agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth'))  # Load the trained weights
agent.local_qnetwork.eval()  # Set the network to evaluation mode

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state, epsilon=0)  # Use epsilon=0 for no exploration
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v3')