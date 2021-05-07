import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import os
import time
import matplotlib.pyplot as plt
import math
import argparse
import torch
from networks.simple_DQN import DQN
import torch.optim as optim
from networks.ReplyMemory import ReplayMemory, Transition
import random
import torch.nn.functional as F
from itertools import count, product
from cores.curiosity_gym import curiosityGym
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='curiosityExplorationEnv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--BATCH_SIZE', type=float, default=256)
    parser.add_argument('--GAMMA', type=float, default=0.999)
    parser.add_argument('--EPS_START', type=float, default=0.9)
    parser.add_argument('--EPS_END', type=float, default=0.05)
    parser.add_argument('--EPS_DECAY', type=float, default=200)
    parser.add_argument('--TARGET_UPDATE', type=float, default=50)

    parser.add_argument('--num_episodes', type=float, default=5000)

    parser.add_argument('--width', type=float, default=128)
    parser.add_argument('--height', type=float, default=128)

    parser.add_argument('--restore_path', type=str, default=None)

    parser.add_argument(
        '--device', type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


steps_done = 0

ACTION_MAP = dict([(tuple(value), key) for key, value in curiosityGym.get_action_info()['action_map'].items()])
INVERSE_ACTION_SPACE = {
    1: 3,
    3: 1,
    2: 4,
    4: 2,
    0: 0
}


def select_action(state, vehicle_status):
    global steps_done
    sample = random.random()
    eps_threshold = args.EPS_END + (args.EPS_START - args.EPS_END) * \
                    math.exp(-1. * steps_done / args.EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        center = [x.item() for x in torch.where(state[0][0]==-2)]
        random_action = []
        for (dx, dy) in list(ACTION_MAP.keys()):
            if dx == dy:
                continue
            elif state[0][0][center[0]+dx][center[0]+dy] == 1:
                random_action += [ACTION_MAP[(dx, dy)]]
        random_action += [INVERSE_ACTION_SPACE[vehicle_status['direction']]]
        random_action = np.random.choice(np.unique(random_action)) - 1
        return torch.tensor([[random_action]], device=args.device, dtype=torch.long)


def optimize_model():
    if len(memory) < args.BATCH_SIZE:
        return
    transitions = memory.sample(args.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=args.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(args.BATCH_SIZE, device=args.device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.GAMMA) + reward_batch
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == '__main__':
    best_score = -math.inf
    args = get_args()
    env = curiosityGym()
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    writer = SummaryWriter('./logs')

    n_actions = env.get_action_info()['action_space']-1

    args.height, args.width = env.get_observation_info()['obs_shape']

    policy_net = DQN(args.height, args.width, 3, n_actions, args.device).to(args.device)
    target_net = DQN(args.height, args.width, 3, n_actions, args.device).to(args.device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    for i_episode in range(args.num_episodes):
        # Initialize the environment and state
        obs = env.reset()
        state = torch.unsqueeze(torch.Tensor(obs), 0)
        # state.requires_grad = True
        # state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, env.get_vehicle_status())
            '''
                JUST using ACTION 1-5
            '''
            obs, reward, done, info = env.step(action[0][0].item()+1)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            next_state = torch.unsqueeze(torch.Tensor(obs), 0)

            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            # print(i_episode, t, action, reward[0].item(), )
            if done or info['loop_detected']:
                score = env.get_score()
                # print(i_episode, 'steps done', steps_done, 'score ', score, 'reward ', reward[0].item(), )
                writer.add_scalar('score', env.score, i_episode)
                if score > best_score:
                    best_score = score
                    print(i_episode, 'steps done', t, 'score ', best_score, 'reward ', reward[0].item(),)
                    torch.save(policy_net.state_dict(), os.path.join(args.logdir, 'policy.pth'))
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % args.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
