import copy
import numpy as np
class Sudoku:
    def __init__(self, board):
        self.board = board
        self.blank_num = 0
        for i in board:
            for j in i:
                if j == 0:
                    self.blank_num += 1
        self.difficulty = int(float(self.blank_num) / 81 * 10)
        self.board_correct = copy.deepcopy(self.board)

    def solve(self):
        self.res_board = copy.deepcopy(self.board_correct)
        self.nums = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.row = [set() for _ in range(9)]
        self.col = [set() for _ in range(9)]
        self.palace = [[set() for _ in range(3)] for _ in range(3)]  # 3*3
        self.blank = []

        # 初始化，按照行、列、宫 分别存入哈希表
        for i in range(9):
            for j in range(9):
                num = self.board_correct[i][j]
                if num == 0:
                    self.blank.append((i, j))
                else:
                    self.row[i].add(num)
                    self.col[j].add(num)
                    self.palace[i//3][j//3].add(num)
        flag = self.dfs(0)
        if flag:
            print('Solve Successfully!')
            return 1
        else:
            print('Wrong Input Board!')
            return 0

    def dfs(self, n):
        if len(self.blank) == 0:
            return True
        min_count = 10
        for blank_pos in self.blank:
            rest = self.nums - self.row[blank_pos[0]] - self.col[blank_pos[1]] - self.palace[blank_pos[0]//3][blank_pos[1]//3]
            if len(rest) > min_count:
                continue
            i, j = blank_pos
            min_count = len(rest)

        rest = self.nums - self.row[i] - self.col[j] - self.palace[i//3][j//3]  # 剩余的数字

        if not rest:
            return False
        for num in rest:
            self.res_board[i][j] = num
            self.row[i].add(num)
            self.col[j].add(num)
            self.palace[i//3][j//3].add(num)
            self.blank.remove((i, j))
            if self.dfs(n+1):
                return True
            self.row[i].remove(num)
            self.col[j].remove(num)
            self.palace[i//3][j//3].remove(num)
            self.blank.append((i, j))
        return False

    def show(self, which_board):
        for i in range(9):
            for j in range(9):
                if which_board == 'original':
                    print(self.board[i][j],end='\t')
                elif which_board == 'result':
                    print(self.res_board[i][j], end='\t')
            print('\n')

    def correct(self):
        self.nums_correct = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.row_correct = [set() for _ in range(9)]
        self.col_correct = [set() for _ in range(9)]
        self.palace_correct = [[set() for _ in range(3)] for _ in range(3)]  # 3*3
        self.board_correct = np.zeros((9, 9), dtype=int)
        self.blank_correct = []
        self.pos_correct = None
        # 初始化，按照行、列、宫 分别存入哈希表
        for i in range(9):
            for j in range(9):
                num = self.board[i][j]
                if num != 0:
                    if (num in self.row_correct[i]) or (num in self.col_correct[j]) or (num in self.palace_correct[i//3][j//3]):
                        self.blank.append((i, j))
                        self.pos_correct = (i, j)
                    else:
                        self.row[i].add(num)
                        self.col[j].add(num)
                        self.palace[i // 3][j // 3].add(num)
                        self.board_correct[i][j] = num
        if self.solve():
            return True
        for i in range(9):
            for j in range(9):
                num = self.board_correct[i][j]
                if num != 0:
                    self.board_correct[i][j] = 0
                    flag = self.solve()
                    if flag:
                        self.pos_correct = (i, j)
                        self.board_correct[i][j] = self.res_board[i][j]
                        return True
                    else:
                        self.board_correct[i][j] = num
        return False

'''
a = np.array([[1, 0, 8, 0, 9, 0, 1, 0, 0],
              [0, 0, 2, 0, 5, 6, 7, 0, 8],
              [6, 0, 0, 0, 0, 3, 0, 4, 0],
              [0, 0, 7, 0, 3, 8, 2, 6, 0],
              [0, 0, 5, 0, 4, 0, 8, 0, 0],
              [0, 6, 4, 0, 7, 0, 9, 0, 0],
              [0, 4, 0, 1, 0, 0, 0, 0, 9],
              [7, 0, 3, 9, 2, 0, 6, 0, 5],
              [0, 0, 1, 0, 6, 0, 4, 0, 0]])
b = Sudoku(a)
b.solve()
b.show('result')
'''