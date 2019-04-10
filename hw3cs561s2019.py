import time
import random


class Config:
    def __init__(self, input_file):
        with open(input_file + '.txt', 'rU') as fp:
            lines = fp.readlines()

            index = 0
            self.GridSize = int(lines[index])
            index += 1

            self.BlockOfInterest = set()
            self.CashGrids = dict()

            # Process Walls
            end_index = index + int(lines[index])
            index += 1
            while index <= end_index:
                row_col = lines[index].split(',')
                row = int(row_col[0]) - 1
                col = int(row_col[1]) - 1

                self.BlockOfInterest.add((row, col))
                index += 1

            # Process Cash Grids
            end_index = index + int(lines[index])
            index += 1
            while index <= end_index:
                row_col_reward = lines[index].split(',')
                row = int(row_col_reward[0]) - 1
                col = int(row_col_reward[1]) - 1

                self.BlockOfInterest.add((row, col))
                self.CashGrids[(row, col)] = float(row_col_reward[2])
                index += 1

            # Get Probability of Move
            self.ProbabilityOfMove = float(lines[index])
            self.ProbabilityOfOtherMove = (1.0 - self.ProbabilityOfMove) * 0.5  # Two other actions possible
            index += 1

            # Get Movement Reward
            self.MovementReward = float(lines[index])
            index += 1

            # Get Discount
            self.Discount = float(lines[index])
            index += 1


class MDPSolver:
    def __init__(self, input_file, max_time):
        config = Config(input_file)
        self.action_space = {
            'U': {
                'cost': config.MovementReward,
                'dir_row': -1,
                'dir_col': 0,
                'p': config.ProbabilityOfMove,
                'pretty': u'\u2191',
                'other':
                    [{
                        'dir_row': -1,
                        'dir_col': -1,
                        'p': config.ProbabilityOfOtherMove
                    },
                        {
                            'dir_row': -1,
                            'dir_col': 1,
                            'p': config.ProbabilityOfOtherMove
                        }]
            },
            'D': {
                'cost': config.MovementReward,
                'dir_row': 1,
                'dir_col': 0,
                'p': config.ProbabilityOfMove,
                'pretty': u'\u2193',
                'other':
                    [{
                        'dir_row': 1,
                        'dir_col': 1,
                        'p': config.ProbabilityOfOtherMove
                    },
                        {
                            'dir_row': 1,
                            'dir_col': -1,
                            'p': config.ProbabilityOfOtherMove
                        }]
            },
            'L': {
                'cost': config.MovementReward,
                'dir_row': 0,
                'dir_col': -1,
                'p': config.ProbabilityOfMove,
                'pretty': u'\u2190',
                'other':
                    [{
                        'dir_row': 1,
                        'dir_col': -1,
                        'p': config.ProbabilityOfOtherMove
                    },
                        {
                            'dir_row': -1,
                            'dir_col': -1,
                            'p': config.ProbabilityOfOtherMove
                        }]
            },
            'R': {
                'cost': config.MovementReward,
                'dir_row': 0,
                'dir_col': 1,
                'p': config.ProbabilityOfMove,
                'pretty': u'\u2192',
                'other':
                    [{
                        'dir_row': -1,
                        'dir_col': 1,
                        'p': config.ProbabilityOfOtherMove
                    },
                        {
                            'dir_row': 1,
                            'dir_col': 1,
                            'p': config.ProbabilityOfOtherMove
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

        self.grid_size = config.GridSize
        boi = config.BlockOfInterest
        self.cash = config.CashGrids
        self.movement_rewards = config.MovementReward
        self.discount = config.Discount

        self.boardA = dict()
        self.boardB = dict()

        self.policy = dict()

        for row in xrange(self.grid_size):
            for col in xrange(self.grid_size):
                key = (row, col)
                self.policy[key] = 'N'
                if key not in boi:
                    self.boardA[key] = 0.0
                    self.boardB[key] = 0.0
                else:
                    if key in self.cash:
                        self.policy[key] = 'E'

        self.end_time = time.time() + max_time

    def get_reward_and_value(self, r, c, move):
        action = self.action_space[move]
        board = self.boardA
        cash = self.cash

        nr = r + action['dir_row']
        nc = c + action['dir_col']
        prob = action['p']
        utility_val = prob * board.get((nr, nc),
                                       cash.get((nr, nc),
                                                board[r, c]))

        other = action['other'][0]
        nr = r + other['dir_row']
        nc = c + other['dir_col']
        prob = other['p']
        utility_val += prob * board.get((nr, nc),
                                        cash.get((nr, nc),
                                                 board[r, c]))

        other = action['other'][1]
        nr = r + other['dir_row']
        nc = c + other['dir_col']
        prob = other['p']
        utility_val += prob * board.get((nr, nc),
                                        cash.get((nr, nc),
                                                 board[r, c]))

        return utility_val

    def update(self, row, col):
        reward_values = [(self.get_reward_and_value(row, col, move), move) for move in self.performable_actions]
        max_reward, max_reward_dir = max(reward_values)

        computed_reward = self.movement_rewards + self.discount * max_reward
        return computed_reward, max_reward_dir

    def do_iteration(self):
        for key, value in self.boardA.iteritems():
            row, col = key
            new_value, action = self.update(row, col)
            self.policy[key] = action
            self.boardB[key] = new_value

    def solve(self):
        #iterations = 0
        #sum_time = 0.0
        while time.time() <= self.end_time:
            #start = time.time()
            self.do_iteration()
            self.boardA, self.boardB = self.boardB, self.boardA
            #end = time.time()

            #print('Took {}'.format(end - start))
            #sum_time += (end - start)
            #iterations += 1

        #print('Average: {}'.format(sum_time / iterations))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        full_str = ''
        for row in xrange(self.grid_size):
            row_str = ''
            for col in xrange(self.grid_size):
                action = self.policy[(row, col)]
                row_str += self.action_space[action]['pretty'] + ' '

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
    solver = MDPSolver('input3', 25.0)

    solver.solve()
    solver.write_out('output3')
    #print(unicode(solver))


if __name__ == '__main__':
    main()