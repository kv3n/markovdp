class Config:
    def __init__(self, input_file):
        with open(input_file, 'rU') as fp:
            lines = fp.readlines()

            index = 0
            self.GridSize = int(lines[index])
            self.Grid = [[0 for _ in xrange(self.GridSize)] for _ in xrange(self.GridSize)]

            index += 1
            self.NumWalls = int(lines[index])
            index += 1
            end_index = index + self.NumWalls
            while index < end_index:
                row_col = lines[index].split(',')
                row = int(row_col[0]) - 1
                col = int(row_col[1]) - 1

                self.Grid[row][col] = -100
                index += 1

            self.NumTerminal = int(lines[index])
            index += 1

            end_index = index + self.NumTerminal
            while index < end_index:
                row_col_reward = lines[index].split(',')
                row = int(row_col_reward[0]) - 1
                col = int(row_col_reward[1]) - 1

                self.Grid[row][col] = int(row_col_reward[2])
                index += 1

            self.P = float(lines[index])

            index += 1
            self.Rp = float(lines[index])

            index += 1
            self.Gamma = float(lines[index])

config = Config('input0.txt')

