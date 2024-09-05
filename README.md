
# Lunar Lander - Deep Q-Network (DQN)

This repository contains the implementation of a Deep Q-Network (DQN) agent that solves the **LunarLander-v2** environment from OpenAI Gym using PyTorch.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Contribution](#contribution)
- [License](#license)

## Introduction

This project implements a DQN agent to solve the **LunarLander-v2** environment using PyTorch and OpenAI Gym. The agent uses experience replay and a target network to improve the stability of the learning process.

The goal is for the agent to land the lunar lander on the target surface as smoothly and efficiently as possible, maximizing the cumulative rewards over episodes.

## Project Structure

```
├── main.py                 # Contains the code
├── README.md               # This readme file
├── checkpoint.pth          # Model weights after training (saved checkpoint)
└── trained_agent_video.mp4 # Video recording of the trained agent
```

## Dependencies

The project requires Python 3.x and the following libraries:

- `torch`
- `numpy`
- `gymnasium`
- `IPython`
- `imageio`
- `random`

You can install all dependencies using the following command:

```bash

pip install numpy gymnasium ipython imageio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade cryptography
pip install --upgrade wheel setuptools
pip install gymnasium[box2d]


```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/lunar-lander-dqn.git
cd lunar-lander-dqn
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, if you don't have a `requirements.txt` file, you can install the necessary dependencies as shown in the Dependencies section.

3. Run the code:

```bash
python main.py
```

## Training

To train the agent from scratch, run:

```bash
python main.py
```

The DQN agent will interact with the environment over multiple episodes, updating its policy based on the experience replay and target networks. The training process will output the progress every 100 episodes and save a checkpoint of the model when the environment is solved (when the average score reaches 200.0 over 100 episodes).

You can adjust the hyperparameters (e.g., learning rate, batch size) by modifying the constants defined in `main.py`.

## Usage

Once trained, the agent can be used to play the Lunar Lander environment. You can load the saved model (`checkpoint.pth`) and watch the agent in action.

To run the agent:

```bash
python main.py 
```

## Results

During the training process, the agent’s performance will be tracked. Once the average score over 100 episodes reaches 200.0, the environment is considered solved, and the model weights will be saved to `checkpoint.pth`.

Sample output:

```
![trained_agent_video-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/f4b2e4a4-e33c-4030-a1ad-c29a5a09471a)
![Screenshot 2024-09-05 220104](https://github.com/user-attachments/assets/aea8f510-bcd1-44a5-b59c-2cb8ffb4597e)

```

## Watch the Agent Play

A pre-trained model has been provided. To see the agent play the game, you can watch the video by running the code or viewing the `video.mp4` file.

You can generate a new video of the trained agent using the following function:

```python
show_video_of_model(agent, 'LunarLander-v2')
```

The video will be saved as `video.mp4` in the project directory.


## Contribution

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
