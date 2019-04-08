import time
import random


class Config:
    def __init__(self, input_file):
        with open(input_file + '.txt', 'rU') as fp:
            lines = fp.readlines()

            index = 0
            self.GridSize = int(lines[index])
            index += 1

            self.BlockOfInterest = dict()

            end_index = index + int(lines[index])
            index += 1
            while index <= end_index:
                row_col = lines[index].split(',')
                row = int(row_col[0]) - 1
                col = int(row_col[1]) - 1

                self.BlockOfInterest[(row, col)] = -100.0
                index += 1

            end_index = index + int(lines[index])
            index += 1
            while index <= end_index:
                row_col_reward = lines[index].split(',')
                row = int(row_col_reward[0]) - 1
                col = int(row_col_reward[1]) - 1

                self.BlockOfInterest[(row, col)] = float(row_col_reward[2])
                index += 1

            self.ProbabilityOfMove = float(lines[index])
            self.ProbabilityOfOtherMove = (1.0 - self.ProbabilityOfMove) * 0.5  # Two other actions possible
            index += 1

            self.MovementReward = float(lines[index])
            index += 1

            self.Discount = float(lines[index])
            index += 1


class MDPSolver:
    def __init__(self, input_file, max_time, stochastic=False):
        self.config = Config(input_file)
        self.action_space = {
            'U': {
                'cost': self.config.MovementReward,
                'dir_row': -1,
                'dir_col': 0,
                'p': self.config.ProbabilityOfMove,
                'pretty': u'\u2191',
                'other':
                    [{
                        'dir_row': -1,
                        'dir_col': -1,
                        'p': self.config.ProbabilityOfOtherMove
                    },
                    {
                        'dir_row': -1,
                        'dir_col': 1,
                        'p': self.config.ProbabilityOfOtherMove
                    }]
            },
            'D': {
                'cost': self.config.MovementReward,
                'dir_row': 1,
                'dir_col': 0,
                'p': self.config.ProbabilityOfMove,
                'pretty': u'\u2193',
                'other':
                    [{
                        'dir_row': 1,
                        'dir_col': 1,
                        'p': self.config.ProbabilityOfOtherMove
                    },
                    {
                        'dir_row': 1,
                        'dir_col': -1,
                        'p': self.config.ProbabilityOfOtherMove
                    }]
            },
            'L': {
                'cost': self.config.MovementReward,
                'dir_row': 0,
                'dir_col': -1,
                'p': self.config.ProbabilityOfMove,
                'pretty': u'\u2190',
                'other':
                    [{
                        'dir_row': 1,
                        'dir_col': -1,
                        'p': self.config.ProbabilityOfOtherMove
                    },
                    {
                        'dir_row': -1,
                        'dir_col': -1,
                        'p': self.config.ProbabilityOfOtherMove
                    }]
            },
            'R': {
                'cost': self.config.MovementReward,
                'dir_row': 0,
                'dir_col': 1,
                'p': self.config.ProbabilityOfMove,
                'pretty': u'\u2192',
                'other':
                    [{
                        'dir_row': -1,
                        'dir_col': 1,
                        'p': self.config.ProbabilityOfOtherMove
                    },
                    {
                        'dir_row': 1,
                        'dir_col': 1,
                        'p': self.config.ProbabilityOfOtherMove
                    }]
            },
            'E': {
                'cost': 0,
                'dir_row': 0,
                'dir_col': 0,
                'p': 1.0,
                'pretty': '$',
                'other': []
            },
            'N': {
                'cost': 0,
                'dir_row': 0,
                'dir_col': 0,
                'p': 1.0,
                'pretty': 'X',
                'other': []
            }
        }

        self.performable_actions = ['U', 'D', 'R', 'L']

        self.grid_size = self.config.GridSize
        boi = self.config.BlockOfInterest

        self.state = dict()
        self.policy = dict()
        self.reward = dict()
        self.blocks = set()
        self.cash_grids = set()
        for row in xrange(self.grid_size):
            for col in xrange(self.grid_size):
                key = (row, col)
                reward = boi.get(key, 0.0)
                self.state[key] = reward
                self.policy[key] = 'N'

                if reward < 0.0:
                    self.blocks.add(key)
                elif reward > 0.0:
                    self.policy[key] = 'E'
                    self.cash_grids.add(key)

        self.end_time = time.time() + max_time

        if stochastic:
            self.update = self.update_stochastic
        else:
            self.update = self.update_all_actions

    def get_value_at(self, nr, nc, r, c, prob):
        value = self.state.get((nr, nc), self.state.get((r, c), 0))

        return prob * value

    def get_reward_and_value(self, row, col, move):
        action = self.action_space[move]
        move_cost = action['cost']

        reward_val = self.get_value_at(nr=row+action['dir_row'],
                                       nc=col+action['dir_col'],
                                       r=row,
                                       c=col,
                                       prob=action['p'])

        for other in action['other']:
            reward_val += self.get_value_at(nr=row+other['dir_row'],
                                            nc=col+other['dir_col'],
                                            r=row,
                                            c=col,
                                            prob=other['p'])

        reward_val = move_cost + self.config.Discount * reward_val

        return reward_val

    def update_all_actions(self, row, col, cur_value, cur_action):
        reward_values = [self.get_reward_and_value(row, col, move) for move in self.performable_actions]

        max_reward = max(reward_values)
        max_reward_dir = [self.performable_actions[idx] for idx, val in enumerate(reward_values) if val == max_reward][0]

        """
        if max_reward < cur_value:
            max_reward = cur_value
            max_reward_dir = cur_action
        """

        return max_reward, max_reward_dir

    def update_stochastic(self, row, col, cur_value, cur_action):
        do_able_actions = list(self.performable_actions)
        if (row, col) in self.cash_grids:
            do_able_actions.append('E')

        moves = [random.choice(do_able_actions)]
        reward_values = [self.get_reward_and_value(row, col, move) for move in moves]

        moves.append(cur_action)
        reward_values.append(cur_value)

        max_reward = max(reward_values)
        max_reward_dir = [moves[idx] for idx, val in enumerate(reward_values) if val == max_reward][0]
        return max_reward, max_reward_dir

    def solve(self):
        converged = False
        while time.time() <= self.end_time and not converged:
            converged = True
            for row in xrange(self.grid_size):
                for col in xrange(self.grid_size):
                    key = (row, col)
                    if key in self.blocks or key in self.cash_grids:
                        continue

                    new_value, action = self.update(row, col, self.state[key], self.policy[key])
                    if abs(self.state[key] - new_value) > 0.0:
                        converged = False
                        self.policy[key] = action
                        self.state[key] = new_value

        if converged:
            print 'Converged with ' + str(self.end_time - time.time()) + ' seconds left'

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        full_str = ''
        for row in xrange(self.grid_size):
            row_str = ''
            for col in xrange(self.grid_size):
                action = self.policy[(row, col)]
                if action == 'N':
                    row_str += 'X ({})'.format(self.state[(row, col)])
                else:
                    row_str += self.action_space[action]['pretty'] + '({}) '.format(self.state[(row, col)])

            full_str += row_str + '\n'

        return full_str

    def write_out(self, output):
        full_str = ''
        for row in xrange(self.grid_size):
            row_str = ''
            for col in xrange(self.grid_size):
                action = self.policy[(row, col)]
                row_str += str(action) + ','
            row_str = row_str.rstrip(',')

            full_str += row_str + '\n'

        full_str = full_str[:-1]

        with open(output + '.txt', 'w') as fp:
            fp.write(full_str)


def main():
    solver = MDPSolver('input3', 30.0, stochastic=False)

    solver.solve()
    solver.write_out('output3')
    #print(unicode(solver))


if __name__ == '__main__':
    main()