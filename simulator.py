from hw3cs561s2019 import *

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Simulator:
    def __init__(self, number):
        self.solver = MDPSolver('input' + str(number), 30.0)
        self.policy = []
        with open('output'+str(number)+'.txt', 'r') as fp:
            for policy_line in fp.readlines():
                in_rows = policy_line.rstrip('\n').split(',')
                policy_row = []
                for policy_col in in_rows:
                    policy_row.append(policy_col)
                self.policy.append(policy_row)

        self.grid_size = len(self.policy)

    def start_at(self, row, col):
        money_made = 0.0
        path = []

        cur_row = row
        cur_col = col
        cur_policy = self.policy[cur_row][cur_col]
        while cur_policy is not 'E':
            path.append((cur_row, cur_col))
            action = self.solver.action_space[cur_policy]
            prob = action['p']
            other_prob = prob + ((1.0 - prob) * 0.5)
            rand_val = random.random()
            new_row = cur_row
            new_col = cur_col
            if rand_val < prob:
                new_row += action['dir_row']
                new_col += action['dir_col']
            elif rand_val < other_prob:
                new_row += action['other'][0]['dir_row']
                new_col += action['other'][0]['dir_col']
            else:
                new_row += action['other'][1]['dir_row']
                new_col += action['other'][1]['dir_col']

            if new_row < 0 or new_row >= self.grid_size:
                new_row = cur_row
                new_col = cur_col
            elif new_col < 0 or new_col >= self.grid_size:
                new_row = cur_row
                new_col = cur_col

            cur_row = new_row
            cur_col = new_col

            money_made += action['cost']

            cur_policy = self.policy[cur_row][cur_col]

            if cur_policy == 'N':
                cur_row = row
                cur_col = col
                path = []

        path.append((cur_row, cur_col))
        money_made += self.solver.state[(cur_row, cur_col)]

        return money_made, path

    def run_simulation(self):
        for row in xrange(self.grid_size):
            for col in xrange(self.grid_size):
                if self.policy[row][col] == 'N':
                    continue

                money_made, path = self.start_at(row, col)
                print_color = bcolors.FAIL if money_made <= 0.0 else bcolors.OKGREEN
                print(print_color + 'Made {} at {}, {} using {}'.format(money_made, row, col, path) + bcolors.ENDC)


def main():
    simulator = Simulator(2)
    simulator.run_simulation()


if __name__ == '__main__':
    main()