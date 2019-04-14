import time
import random

index_to_actions_map = ['U', 'D', 'L', 'R']
actions_to_index_map = {
    'U': 0,
    'D': 1,
    'L': 2,
    'R': 3
}

class Treasure:
    def __init__(self, value, prob, other_prob):
        self.policy = 'E'
        self.utility = [prob * value, other_prob * value]


class Wall:
    def __init__(self):
        self.policy = 'N'


class Location:
    def __init__(self, prob, other_prob, movement_reward, discount, max_payout):
        self.prob = prob
        self.other_prob = other_prob
        self.movement_reward = movement_reward
        self.discount = discount

        # Actions: U D L R
        self.actions = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        
        self.utility = [0.0, 0.0]
        self.policy = random.choice(index_to_actions_map)

    def do_update(self):
        a = self.actions

        utilities = [(a[0][0][0] + a[0][1][1] + a[0][2][1], 'U'),
                     (a[1][0][0] + a[1][1][1] + a[1][2][1], 'D'),
                     (a[2][0][0] + a[2][1][1] + a[2][2][1], 'L'),
                     (a[3][0][0] + a[3][1][1] + a[3][2][1], 'R')]

        max_utility, self.policy = max(utilities)
        max_utility = self.movement_reward + self.discount * max_utility

        cur_utility = self.utility[0]

        self.utility[0] = self.prob * max_utility
        self.utility[1] = self.other_prob * max_utility

        return abs(self.utility[0] - cur_utility) < 0.00001
    
    def evaluate(self):
        a = self.actions[actions_to_index_map[self.policy]]
        random_policy_utility = a[0][0] + a[1][1] + a[2][1]
        
        self.utility[0] = self.prob * random_policy_utility
        self.utility[1] = self.other_prob * random_policy_utility


def read_file(input_file):
    cashes = dict()
    walls = dict()

    max_payout = 0.0

    with open(input_file + '.txt', 'rU') as fp:
        lines = fp.readlines()

        index = 0
        grid_size = int(lines[index])
        index += 1

        # Process Walls
        end_index = index + int(lines[index])
        index += 1
        while index <= end_index:
            row_col = lines[index].split(',')
            row = int(row_col[0]) - 1
            col = int(row_col[1]) - 1

            walls[(row, col)] = Wall()
            index += 1

        # Process Cash Grids
        end_index = index + int(lines[index])
        index += 1

        # get prob and other prob
        prob = float(lines[end_index + 1])
        other_prob = (1.0 - prob) * 0.5  # Two other actions possible

        while index <= end_index:
            row_col_reward = lines[index].split(',')
            row = int(row_col_reward[0]) - 1
            col = int(row_col_reward[1]) - 1
            payout = float(row_col_reward[2])

            cashes[(row, col)] = Treasure(payout, prob, other_prob)
            index += 1

            max_payout = max(max_payout, payout)

        # We already fetched prob and other prob. So move iterator
        index += 1

        # Get Movement Reward
        reward = float(lines[index])
        index += 1

        # Get Discount
        discount = float(lines[index])
        index += 1

    return grid_size, cashes, walls, reward, discount, prob, other_prob, max_payout


class MDPSolver:
    def __init__(self, input_file, max_time):
        self.start_time = time.time()
        self.end_time = self.start_time + max_time

        self.grid_size, self.cashes, self.walls, reward, discount, prob, other_prob, max_payout = read_file(input_file)

        board = dict()

        for row in xrange(self.grid_size):
            for col in xrange(self.grid_size):
                key = (row, col)
                if key not in self.cashes and key not in self.walls:
                    board[key] = Location(prob, other_prob, reward, discount, max_payout)

        # Cache locations
        for key, location in board.iteritems():
            r, c = key

            # Up
            nr = r - 1
            nc = c
            location.actions[0][0] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r - 1
            nc = c - 1
            location.actions[0][1] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r - 1
            nc = c + 1
            location.actions[0][2] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            # Down
            nr = r + 1
            nc = c
            location.actions[1][0] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r + 1
            nc = c - 1
            location.actions[1][1] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r + 1
            nc = c + 1
            location.actions[1][2] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            # Left
            nr = r
            nc = c - 1
            location.actions[2][0] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r - 1
            nc = c - 1
            location.actions[2][1] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r + 1
            nc = c - 1
            location.actions[2][2] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            # Right
            nr = r
            nc = c + 1
            location.actions[3][0] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r - 1
            nc = c + 1
            location.actions[3][1] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

            nr = r + 1
            nc = c + 1
            location.actions[3][2] = board.get((nr, nc), self.cashes.get((nr, nc), board[key])).utility

        self.resting_board = board
        self.updating_board = list(board.values())
    
    def evaluate(self, times):
        for _ in xrange(times):
            for location in self.updating_board:
                location.evaluate()

    def solve(self):
        num_updating_boards = len(self.updating_board)
        min_batching = int(self.grid_size * 0.75)
        max_batching = int(self.grid_size * 1.25)
        end_index = num_updating_boards - min_batching
        iterations = 0
        converge_times = 0

        start = time.time()
        
        # Evaluate k times
        self.evaluate(times=2)
        
        # Run random subset based iteration
        while time.time() <= self.end_time:
            if iterations % num_updating_boards == 0:
                print('Shuffled')
                random.shuffle(self.updating_board)

            converged = True
            start_idx = random.randint(0, end_index)
            cur_batching = random.randint(min_batching, max_batching)
            for location in self.updating_board[start_idx:start_idx+cur_batching]:
                converged = converged & location.do_update()
                location.do_update()

            if converged:
                converge_times += 1
                if converge_times > 10:
                    print('Converged')
                    break
            else:
                converge_times = 0

            iterations += 1

        end = time.time()

        print('Average: {} over {} iterations'.format((end - start) / iterations, iterations))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        pretty_dict = {
            'U': u'\u2191',
            'D': u'\u2193',
            'L': u'\u2190',
            'R': u'\u2192',
            'N': 'X',
            'E': '$'
        }
        full_str = ''
        for row in xrange(self.grid_size):
            row_str = ''
            for col in xrange(self.grid_size):
                key = (row, col)
                action = self.resting_board.get(key, self.cashes.get(key, self.walls.get(key, None))).policy
                row_str += pretty_dict[action] + ' '

            full_str += row_str + '\n'

        return full_str

    def write_out(self, output):
        full_str = ''
        for row in xrange(self.grid_size):
            row_str = ''
            for col in xrange(self.grid_size):
                key = (row, col)
                action = self.resting_board.get(key, self.cashes.get(key, self.walls.get(key, None))).policy
                row_str += action + ','
            row_str = row_str.rstrip(',')

            full_str += row_str + '\n'

        full_str = full_str[:-1]

        with open(output + '.txt', 'w') as fp:
            fp.write(full_str)


def main():
    solver = MDPSolver('input', 28.0)

    solver.solve()
    solver.write_out('output')
    print(unicode(solver))


if __name__ == '__main__':
    main()
