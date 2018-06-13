import tkinter as tk
import random
import time

# try:
import pygame as pg
# except ImportError:
#     audio = None
# else:
#     audio = True
from matrix_rotation import rotate_array as rotate
import numpy as np


class Shape:
    def __init__(self, shape, key, piece, row, column, coords):
        self.shape = shape
        self.key = key
        self.piece = piece
        self.row = row
        self.column = column
        self.coords = coords


class Tetris(tk.Tk):

    # reset env
    def reset(self):
        self.tickrate = 1000
        self.score = 0
        self.pieceIsActive = False
        self.paused = False

        self.pieces = 0
        self.reward = 0
        self.last_holes = 0
        self.new_holes = 0
        self.last_pileHeight = 0
        self.new_pileHeight = 0
        self.altitudeDifference = 0
        self.empty = 0
        self.rowFull = 0

        self.done = False
        self.draw_board()
        self.spawn()
        # (self.new_holes, self.new_pileHeight, self.altitudeDifference,one hot)
        return self.getState()

    def getStateActionSize(self):
        return 27, 40  # 244, 40

    # it is for get command from RL Agent
    # action is between 0 and 39
    def step(self, action):
        shift_idx = action // 4  # 0<action<9
        rotation_idx = action % 4  # 0<action<3

        # move shapes base on rotation 
        if (rotation_idx == 2):  # 180 =90 * 2
            self.rotate(dir=1)
            self.rotate(dir=1)
        elif (rotation_idx != 0):
            self.rotate(dir=rotation_idx)

        # move shapes base on  shift indexes

        offset = shift_idx - self.activePiece.column
        if (offset > 0):
            for _ in range(offset): self.shift(dir='r')
        else:
            for _ in range(abs(offset)): self.shift(dir='l')

        self.snap()

        self.status = self.getState(), self.getReward(), self.done
        self.setLabels()
        # ready next piece
        if (not self.done):
            self.spawn()
        # print(self.status)
        return self.status

        # compute state after each settle

    def getState(self):
        self.pieces += 1
        tBoard = np.transpose(self.board)

        # Holes
        self.last_holes = self.new_holes
        self.new_holes = 0
        for row in tBoard:
            for col in range(len(row)):
                if row[col] == 'x':
                    col += 1
                    while (col < 24):
                        if row[col] == '':
                            self.new_holes += 1
                        col += 1
                    break

        # Pile Height: tallest column (article 6 page:4)
        height_idx = []
        for idx, row in enumerate(tBoard):
            for idy, col in enumerate(row):
                if col == 'x':
                    height_idx.append(24 - idy)
                    break
                elif idy == 23:
                    height_idx.append(0)

        self.empty = 0
        for i in range(4, 24):
            for j in range(10):
                if self.board[i][j] == '':
                    self.empty += 1

        # full rows
        self.rowFull = 0
        for row in self.board:
            if all(row):
                self.rowFull += 1

        self.empty -= self.new_holes  # empty is not hole
        self.last_pileHeight = self.new_pileHeight
        self.new_pileHeight = np.amax(height_idx)

        # Altitude Difference: tallest column - smallest column (article 6 page:5)
        self.altitudeDifference = self.new_pileHeight - np.amin(height_idx)

        self.inputBoard = []
        for r in self.board:
            for c in r:
                if c == '':
                    self.inputBoard.append(0)
                else:
                    self.inputBoard.append(1)
        self.state = (
            self.new_holes, self.new_pileHeight, self.altitudeDifference, self.empty, *self.oneHot)  # *self.inputBoard)
        # print(self.state)
        return self.state

    def oneHotEncoder(self, shape, rotation):
        self.oneHot = np.zeros(23)
        if shape == 'o':
            rotation = 0
        elif shape == 'I':
            if rotation == 180:
                rotation = 0
            elif rotation == 270:
                rotation = 90

        rot = {0: 0, 90: 1, 180: 2, 270: 3}
        choices = {'s': [0, 1, 2, 3], 'z': [4, 5, 6, 7], 'r': [8, 9, 10, 11], 'L': [12, 13, 14, 15], 'o': [16],
                   'I': [17, 18], 'T': [19, 20, 21, 22]}
        self.oneHot[choices.get(shape)[rot.get(rotation)]] = 1

    def getReward(self):
        self.reward = (self.empty) * -0.125 + (self.pieces) * 5 + (self.new_holes - self.last_holes) * -.1 + (
                self.new_pileHeight - self.last_pileHeight) * -5 + self.rowFull * 100
        return self.reward

    def setLabels(self):
        self.pieces_var.set('Pieces:{}'.format(self.pieces))
        self.holes_var.set('Holes:{}'.format(self.new_holes))
        self.empty_var.set('Empty:{}'.format(self.empty))
        self.pileHeight_var.set('PileHeight: {}'.format(self.new_pileHeight))
        self.altitudeDifference_var.set('Altitude Difference: {}'.format(self.altitudeDifference))

    def __init__(self, parent, render):

        parent.title('RL Tetris')
        self.parent = parent
        self.render = render  # show game board
        self.board_width = 10
        self.board_height = 24
        self.width = 400
        self.height = 960
        self.high_score = 0
        self.square_width = self.width // 10

        self.shapes = {
            's': [['*', ''],
                  ['*', '*'],
                  ['', '*']],

            'z': [['', '*'],
                  ['*', '*'],
                  ['*', '']],

            'r': [['*', '*'],
                  ['*', ''],
                  ['*', '']],

            'L': [['*', ''],
                  ['*', ''],
                  ['*', '*']],

            'o': [['*', '*'],
                  ['*', '*']],

            'I': [['*'],
                  ['*'],
                  ['*'],
                  ['*']],

            'T': [['*', '*', '*'],
                  ['', '*', '']]
        }
        self.colors = {'s': 'green', 'z': 'yellow', 'r': 'turquoise', 'L': 'orange', 'o': 'blue', 'I': 'red',
                       'T': 'violet'}

        for key in ('<Down>', '<Left>', '<Right>', 'a', 'A', 's', 'S', 'd', 'D'):
            self.parent.bind(key, self.shift)

        for key in ('0', 'q', 'Q', 'e', 'E', '<Up>', 'w', 'W'):
            self.parent.bind(key, self.rotate)

        self.parent.bind('<space>', self.snap)
        self.parent.bind('<Escape>', self.pause)
        self.parent.bind('n', self.reset)
        self.parent.bind('N', self.reset)
        self.parent.bind('g', self.toggle_guides)
        self.parent.bind('G', self.toggle_guides)

        self.canvas = None
        self.preview_canvas = None
        self.ticking = None
        self.spawning = None
        self.guide_fill = ''

        self.pieces_var = tk.StringVar()
        self.holes_var = tk.StringVar()
        self.empty_var = tk.StringVar()
        self.pileHeight_var = tk.StringVar()
        self.altitudeDifference_var = tk.StringVar()

        self.pieces_label = tk.Label(parent, textvariable=self.pieces_var, width=20, height=1, font=('Tahoma', 12))
        self.pieces_label.grid(row=1, column=1, sticky='N')  # sticky: N S E W

        self.holes_label = tk.Label(parent, textvariable=self.holes_var, width=20, height=1, font=('Tahoma', 12))
        self.holes_label.grid(row=2, column=1, sticky='N')  # sticky: N S E W

        self.empty_label = tk.Label(parent, textvariable=self.empty_var, width=20, height=1, font=('Tahoma', 12))
        self.empty_label.grid(row=3, column=1, sticky='N')  # sticky: N S E W

        self.pileHeight_label = tk.Label(parent, textvariable=self.pileHeight_var, width=20, height=1,
                                         font=('Tahoma', 12))
        self.pileHeight_label.grid(row=4, column=1, sticky='N')

        self.altitudeDifference_label = tk.Label(parent, textvariable=self.altitudeDifference_var, width=20, height=1,
                                                 font=('Tahoma', 12))
        self.altitudeDifference_label.grid(row=5, column=1, sticky='N')

    def toggle_guides(self, event=None):
        self.guide_fill = '' if self.guide_fill else 'black'
        self.canvas.itemconfig(self.guides[0], fill=self.guide_fill)
        self.canvas.itemconfig(self.guides[1], fill=self.guide_fill)

    def draw_board(self):
        if self.spawning:
            self.parent.after_cancel(self.spawning)
        self.board = [['' for column in range(self.board_width)] for row in range(self.board_height)]
        self.field = [[None for column in range(self.board_width)] for row in range(self.board_height)]
        if self.canvas:
            self.canvas.destroy()
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0, rowspan=20)
        self.h_separator = self.canvas.create_line(0, self.height // 6, self.width, self.height // 6, width=2)
        self.v_separator = self.canvas.create_line(self.width, 0, self.width, self.height, width=2)
        for c in range(self.board_width):  # col lines
            self.canvas.create_line(c * self.square_width, 0, c * self.square_width, self.height, fill='lightgray')
        for r in range(self.board_height):  # row lines
            self.canvas.create_line(0, r * self.square_width, self.width, r * self.square_width, fill='lightgray')

        if self.preview_canvas:
            self.preview_canvas.destroy()
        self.preview_canvas = tk.Canvas(self.parent, width=5 * self.square_width, height=5 * self.square_width)
        self.preview_canvas.grid(row=0, column=1)

        self.setLabels()

        self.guides = [
            self.canvas.create_line(0, 0,
                                    0, self.height),
            self.canvas.create_line(self.width, 0,
                                    self.width, self.height)]

    def pause(self, event=None):
        if self.pieceIsActive and not self.paused:
            self.paused = True
            self.pieceIsActive = False
            self.parent.after_cancel(self.ticking)
        elif self.paused:
            self.paused = False
            self.pieceIsActive = True
            self.ticking = self.parent.after(self.tickrate, self.tick)

    def print_board(self):
        for row in self.board:
            print(*(cell or ' ' for cell in row), sep='')

    def check(self, shape, r, c, l, w):
        # check whether we may rotate a piece
        for row, squares in zip(range(r, r + l), shape):
            for column, square in zip(range(c, c + w), squares):
                if (row not in range(self.board_height) or
                    column not in range(self.board_width)) or (
                        square and self.board[row][column] == 'x'):
                    return
        return True

    def move(self, shape, r, c, l, w):
        square_idxs = iter(range(4))
        # remove shape from board
        for row in self.board:
            row[:] = ['' if cell == '*' else cell for cell in row]
        # put shape onto board
        for row, squares in zip(range(r, r + l), shape):
            for column, square in zip(range(c, c + w), squares):
                if square:
                    self.board[row][column] = square
                    square_idx = next(square_idxs)
                    coord = (column * self.square_width, row * self.square_width,
                             (column + 1) * self.square_width, (row + 1) * self.square_width)
                    self.activePiece.coords[square_idx] = coord
                    self.canvas.coords(self.activePiece.piece[square_idx], coord)

        self.activePiece.row = r
        self.activePiece.column = c
        self.activePiece.shape = shape
        self.move_guides(c, c + w)
        # self.print_board()
        return True

    def check_and_move(self, shape, r, c, l, w):
        return self.check(shape, r, c, l, w) and self.move(shape, r, c, l, w)

    def rotate(self, event=None, dir=None):
        if not self.pieceIsActive:
            return
        if len(self.activePiece.shape) == len(self.activePiece.shape[0]):
            return
        r = self.activePiece.row
        c = self.activePiece.column
        l = len(self.activePiece.shape)
        w = len(self.activePiece.shape[0])
        x = c + w // 2  # center column for old shape
        y = r + l // 2  # center row for old shape
        direction = event.keysym if event != None else ''
        if dir == 3 or direction in {'q'}:
            shape = rotate(self.activePiece.shape, -90)
            rotation_index = (self.activePiece.rotation_index - 1) % 4
            ra, rb = self.activePiece.rotation[rotation_index]
            rotation_offsets = -ra, -rb
        elif dir == 1 or direction in {'e'}:
            shape = rotate(self.activePiece.shape, 90)
            rotation_index = self.activePiece.rotation_index
            rotation_offsets = self.activePiece.rotation[rotation_index]
            rotation_index = (rotation_index + 1) % 4

        l = len(shape)  # length new shape
        w = len(shape[0])  # width new shape
        rt = y - l // 2  # row new shape
        ct = x - w // 2  # column new shape
        x_correction, y_correction = rotation_offsets
        rt += y_correction
        ct += x_correction

        if not self.check_and_move(shape, rt, ct, l, w):
            return
        self.activePiece.rotation_index = rotation_index

    def shift(self, event=None, dir=None):
        if not self.pieceIsActive:
            return
        r = self.activePiece.row
        c = self.activePiece.column
        l = len(self.activePiece.shape)
        w = len(self.activePiece.shape[0])
        direction = (event and event.keysym) or 'Down'
        if dir == 'd' or direction in {'s'}:
            rt = r + 1  # row temp
            ct = c  # column temp

        elif dir == 'l' or direction in {'a'}:
            rt = r
            ct = c - 1
        elif dir == 'r' or direction in {'d'}:
            rt = r
            ct = c + 1
        success = self.check_and_move(self.activePiece.shape, rt, ct, l, w)

        if dir == 'd' and not success:
            self.settle()

    def settle(self):

        self.pieceIsActive = False
        for row in self.board:
            row[:] = ['x' if cell == '*' else cell for cell in row]

        for (x1, y1, x2, y2), id in zip(self.activePiece.coords, self.activePiece.piece):
            self.field[y1 // self.square_width][x1 // self.square_width] = id

        self.setLabels()

        if any(any(row) for row in self.board[:4]):
            self.lose()

    def lose(self):
        self.pieceIsActive = False
        self.high_score = max(self.score, self.high_score)
        # self.holes_var.set('Holes:\n{}'.format(self.new_holes))
        self.done = True
        # self.reset()

    def preview(self):
        self.preview_canvas.delete(tk.ALL)
        key = random.choice('szrLoIT')
        rot = random.choice((0, 90, 180, 270))
        shape = rotate(self.shapes[key], rot)
        width = len(shape[0])
        start = 0
        self.previewPiece = Shape(shape, key, [], 0, start, [])
        half = self.square_width // 2
        for y, row in enumerate(shape):
            self.board[y][start:start + width] = shape[y]
            for x, cell in enumerate(row):
                if cell:
                    self.previewPiece.coords.append(
                        (
                            self.square_width * x + half,
                            self.square_width * y + half,
                            self.square_width * (x + 1) + half,
                            self.square_width * (y + 1) + half
                        )
                    )

                    # self.previewPiece.piece.append(
                    #     self.preview_canvas.create_rectangle(
                    #         self.previewPiece.coords[-1], fill=self.colors[key], width=2,
                    #         # outline='dark' + self.colors[key]
                    #     )
                    # )
        ls = len(shape)
        ws = len(shape[0])
        self.previewPiece.rotation_index = 0

        if 3 in (ws, ls):
            self.previewPiece.rotation = [(0, 0), (1, 0), (-1, 1), (0, -1)]
        else:
            self.previewPiece.rotation = [(1, -1), (0, 1), (0, 0), (-1, 0)]

        if (ls < ws):  # wide shape
            self.previewPiece.rotation_index += 1

    def move_guides(self, left, right):
        left *= self.square_width
        right *= self.square_width
        self.canvas.coords(self.guides[0], left, 0, left, self.height)
        self.canvas.coords(self.guides[1], right, 0, right, self.height)

    def spawn(self):

        key = random.choice('szrLoIT')
        rot = random.choice((0, 90, 180, 270))
        self.oneHotEncoder(key, rot)
        shape = rotate(self.shapes[key], rot)
        width = len(shape[0])
        start = (10 - width) // 2
        self.activePiece = Shape(shape, key, [], 0, start, [])

        self.pieceIsActive = True

        self.activePiece.column = start
        self.activePiece.start = start
        self.activePiece.coords = []
        self.activePiece.piece = []
        for y, row in enumerate(self.activePiece.shape):
            if self.activePiece.shape[y] == ('*', '*', '*', '*'):
                tmp = y + 1
                self.activePiece.row = 1
            else:
                tmp = y  # one row down for rotation

            self.board[tmp][start:start + width] = self.activePiece.shape[y]
            for x, cell in enumerate(row, start=start):
                if cell:
                    self.activePiece.coords.append(
                        (
                            self.square_width * x,
                            self.square_width * tmp,
                            self.square_width * (x + 1),
                            self.square_width * (tmp + 1)
                        )
                    )

                    self.activePiece.piece.append(
                        self.canvas.create_rectangle(
                            self.activePiece.coords[-1], fill=self.colors[self.activePiece.key], width=2,
                            # outline='dark' + self.colors[key]
                        )
                    )
        ls = len(self.activePiece.shape)
        ws = len(self.activePiece.shape[0])
        self.activePiece.rotation_index = 0

        if 3 in (ws, ls):
            self.activePiece.rotation = [(0, 0), (1, 0), (-1, 1), (0, -1)]
        else:
            self.activePiece.rotation = [(1, -1), (0, 1), (0, 0), (-1, 0)]

        if (ls < ws):  # wide shape
            self.activePiece.rotation_index += 1

        self.move_guides(start, (start + width))

    def snap(self, dir=None, event=None):
        if not self.pieceIsActive:
            return
        r = self.activePiece.row
        c = self.activePiece.column
        l = len(self.activePiece.shape)
        w = len(self.activePiece.shape[0])

        while self.check(self.activePiece.shape, r + 1, c, l, w):
            r += 1
        self.move(self.activePiece.shape, r, c, l, w)
        self.settle()

    def clear(self, indices):
        # clear filled row and add blank row in top
        for idx in indices:
            self.board.pop(idx)
            self.board.insert(0, ['' for column in range(self.board_width)])
        self.clear_iter(indices)

    def clear_iter(self, indices, current_column=0):
        for row in indices:
            id = self.field[row][current_column]
            self.field[row][current_column] = None
            self.canvas.delete(id)
        if current_column < self.board_width - 1:
            self.clear_iter(indices, current_column + 1)
        else:
            for idx, row in enumerate(self.field):
                offset = sum(r > idx for r in indices) * self.square_width
                for square in row:
                    if square:
                        self.canvas.move(square, 0, offset)

            for row in indices:
                self.field.pop(row)
                self.field.insert(0, [None for x in range(self.board_width)])

# root = tk.Tk()
# root.geometry('+%d+%d' % (800, 10))
# tetris = Tetris(root, render=True)

# state = tetris.reset()
# action = random.randint(0, 39)
# state = tetris.step(action)
# root.mainloop()
