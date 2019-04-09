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
                self.CashGrids[(row, col)] = int(row_col_reward[2])
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
        cash = self.config.CashGrids

        self.boardA = dict()
        self.boardB = dict()

        self.policy = dict()
        self.reward = dict()

        for row in xrange(self.grid_size):
            for col in xrange(self.grid_size):
                key = (row, col)
                self.policy[key] = 'N'
                if key not in boi:
                    self.boardA[key] = 0.0
                    self.boardB[key] = 0.0
                else:
                    if key in cash:
                        self.policy[key] = 'E'
                        self.boardA[key] = float(cash[key])
                        self.boardB[key] = float(cash[key])

        self.end_time = time.time() + max_time

        self.update = self.update_all_actions

    def get_value_at(self, nr, nc, r, c, prob, board):
        value = board.get((nr, nc), board[(r, c)])

        return prob * value

    def get_reward_and_value(self, row, col, move, board):
        action = self.action_space[move]

        utility_val = self.get_value_at(nr=row + action['dir_row'],
                                        nc=col + action['dir_col'],
                                        r=row,
                                        c=col,
                                        prob=action['p'],
                                        board=board)

        for other in action['other']:
            utility_val += self.get_value_at(nr=row + other['dir_row'],
                                             nc=col + other['dir_col'],
                                             r=row,
                                             c=col,
                                             prob=other['p'],
                                             board=board)

        final_val = utility_val

        return final_val

    def update_all_actions(self, row, col, board):
        reward_values = [self.get_reward_and_value(row, col, move, board) for move in self.performable_actions]

        max_reward = max(reward_values)
        maxes = [self.performable_actions[idx] for idx, val in enumerate(reward_values) if val == max_reward]
        max_reward_dir = maxes[0]

        computed_reward = self.config.MovementReward + self.config.Discount * max_reward
        return computed_reward, max_reward_dir

    def do_iteration(self, use_board, update_board):
        converged = True
        for row, col in use_board:
            key = (row, col)
            if key in self.config.CashGrids:
                continue

            cur_value = use_board[key]
            new_value, action = self.update(row, col, use_board)
            self.policy[key] = action
            update_board[key] = new_value

            if converged and abs(cur_value - new_value) > 0.0:
                converged = False

        return converged

    def solve(self):
        use_board = self.boardA
        update_board = self.boardB
        while time.time() <= self.end_time:
            if self.do_iteration(use_board, update_board):
                # print('Converged with {} seconds left'.format(self.end_time - time.time()))
                break

            use_board, update_board = update_board, use_board

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        full_str = ''
        for row in xrange(self.grid_size):
            row_str = ''
            for col in xrange(self.grid_size):
                action = self.policy[(row, col)]
                if action == 'N':
                    row_str += 'X '
                else:
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
    solver = MDPSolver('input', 25.0)

    solver.solve()
    solver.write_out('output')
    # print(unicode(solver))


if __name__ == '__main__':
    main()