import time
import random


class Treasure:
    def __init__(self, value, prob, other_prob):
        self.policy = 'E'
        self.utility = {
            False: [prob * value, other_prob * value],
            True: [prob * value, other_prob * value]
        }


class Wall:
    def __init__(self):
        self.policy = 'N'


class Location:
    def __init__(self, prob, other_prob, movement_reward, discount):
        self.prob = prob
        self.other_prob = other_prob
        self.movement_reward = movement_reward
        self.discount = discount

        self.actions = []
        self.actions.append([None, None, None])  # 'U'
        self.actions.append([None, None, None])  # 'D'
        self.actions.append([None, None, None])  # 'L'
        self.actions.append([None, None, None])  # 'R'

        self.utility = {
            False: [prob * 0.0, other_prob * 0.0],
            True: [prob * 0.0, other_prob * 0.0]
        }
        self.policy = 'N'

    def do_update(self, bit, anti):
        a = self.actions

        utilities = [(a[0][0][bit][0] + a[0][1][bit][1] + a[0][2][bit][1], 'U'),
                     (a[1][0][bit][0] + a[1][1][bit][1] + a[1][2][bit][1], 'D'),
                     (a[2][0][bit][0] + a[2][1][bit][1] + a[2][2][bit][1], 'L'),
                     (a[3][0][bit][0] + a[3][1][bit][1] + a[3][2][bit][1], 'R')]

        max_utility, self.policy = max(utilities)
        max_utility = self.movement_reward + self.discount * max_utility

        self.utility[anti][0] = self.prob * max_utility
        self.utility[anti][1] = self.other_prob * max_utility


def read_file(input_file):
    cashes = dict()
    walls = dict()

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

            cashes[(row, col)] = Treasure(float(row_col_reward[2]), prob, other_prob)
            index += 1

        # We already fetched prob and other prob. So move iterator
        index += 1

        # Get Movement Reward
        reward = float(lines[index])
        index += 1

        # Get Discount
        discount = float(lines[index])
        index += 1

    return grid_size, cashes, walls, reward, discount, prob, other_prob


class MDPSolver:
    def __init__(self, input_file, max_time):
        self.start_time = time.time()
        self.end_time = self.start_time + max_time

        self.grid_size, self.cashes, self.walls, reward, discount, prob, other_prob = read_file(input_file)

        board = dict()

        for row in xrange(self.grid_size):
            for col in xrange(self.grid_size):
                key = (row, col)
                if key not in self.cashes and key not in self.walls:
                    board[key] = Location(prob, other_prob, reward, discount)

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

    def solve(self):
        iterations = 0
        sum_time = 0.0

        bit = False
        anti = not bit
        while time.time() <= self.end_time:
            start = time.time()

            for location in self.updating_board:
                location.do_update(bit, anti)

            bit = not bit
            anti = not bit

            end = time.time()

            print('Took {}'.format(end - start))

            sum_time += (end - start)
            iterations += 1

        print('Average: {} over {} iterations'.format(sum_time / iterations, iterations))

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
    solver = MDPSolver('input3', 28.0)

    solver.solve()
    solver.write_out('output3')
    print(unicode(solver))


if __name__ == '__main__':
    main()
