import tkinter as tk
import random
import time

# try:
import pygame as pg
# except ImportError:
#     audio = None
# else:
#     audio = True
from numpy.core.multiarray import dtype
from tensorflow.python.keras import activations
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
    def reset(self, weights):

        self.w1 = weights[0]  # holes weight
        self.w2 = weights[1]  # var height weight
        self.w3 = weights[2]  # piece sides weight
        self.w4 = weights[3]  # piece height weight

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
        self.rowsFilled = 0
        self.blocksNxN = np.zeros(10)
        self.lowerBlocks = 0
        self.piece_height = []
        self.height_idx = []
        self.done = False
        self.draw_board()
        self.spawn()
        # (self.new_holes, self.new_pileHeight, self.altitudeDifference,one hot)
        return self.getState()

    def getStateActionSize(self):
        # self.reset()
        return 240, 40  # 240, 40

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
        # if self.render:
        #     self.setLabels()
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
        self.piece_holes = self.new_holes - self.last_holes
        # Pile Height: tallest column (article 6 page:4)
        self.height_idx.clear()
        for idx, row in enumerate(tBoard):
            for idy, col in enumerate(row):
                if col == 'x':
                    self.height_idx.append(24 - idy)
                    break
                elif idy == 23:
                    self.height_idx.append(0)

        # self.empty = 0
        # self.lowerBlocks = 0
        # for i in range(4, 24):
        #     col = 0
        #     for j in range(10):
        #         if self.board[i][j] == '':
        #             self.empty += 1
        #         else:
        #             col += 1
        #     self.lowerBlocks = self.lowerBlocks + (col * i)

        # self.full = 0
        #
        # for i in range(4, 24):
        #     for j in range(10):
        #         if self.board[i][j] != '':
        #             self.full += 1

        # self.full/=4 # each shape have 4 pieces
        # self.empty -= self.new_holes  # empty is not hole
        # self.last_pileHeight = self.new_pileHeight
        # self.new_pileHeight = np.amax(height_idx)
        # print(height_idx[self.activePiece.column:len(self.activePiece.shape[0])+1])
        # print(self.activePiece.column,len(self.activePiece.shape[0])+self.activePiece.column+1)
        # self.increase = self.last_pileHeight -\
        #                 np.amax(height_idx[self.activePiece.column:len(self.activePiece.shape[0])+self.activePiece.column+1])
        # self.increase= self.last_pileHeight -np.amax(height_idx[self.activePiece.column:len(self.activePiece.shape[0])+self.activePiece.column+1])#/self.full
        # print(self.increase)
        # self.increase=self.increase/self.full if self.increase<0 else self.increase*self.full
        # self.increase=-np.var(height_idx)+(self.full/4)
        # Altitude Difference: tallest column - smallest column (article 6 page:5)
        # self.altitudeDifference = self.new_pileHeight - np.amin(height_idx)

        # convert x * '' to 0,1
        self.oneHotBoard = []

        for r in self.board:
            for c in r:
                if c == '':
                    self.oneHotBoard = np.append(self.oneHotBoard, 0)
                else:
                    self.oneHotBoard = np.append(self.oneHotBoard, 1)

        # find  rows filled
        # self.lastRowsFilled = self.rowsFilled
        # self.rowsFilled = 0
        # for row in self.board:
        #     if all(c == 'x' for c in row):
        #         self.rowsFilled = self.rowsFilled + 1
        # print(self.oneHotBoard)
        # find rectangle shapes
        # self.calculated_pixel = np.zeros([24, 10])
        # self.blocksNxN = np.zeros(10).astype(np.int8)  # 0 : 10 squares
        # self.inputBoard = np.reshape(self.oneHotBoard, [24, 10])

        # for r in range(4, 24):  # 4:23
        #     for c in range(10):  # 0:9
        #         for square in range(2, 11):  # 3:10 squares
        #             if r + square < 25 and c + square < 11 and self.board[r][c] != '' \
        #                     and self.calculated_pixel[r][c] != 1:
        #                 # print(square, self.inputBoard[r:r + square , c:c + square ])
        #                 if np.all(self.inputBoard[r:r + square, c:c + square]):  # begin:end-1
        #                     self.blocksNxN[square] += 1
        #                     # self.lowerBlocks = self.lowerBlocks + (square * r)
        #                     # self.calculated_pixel[r:r + square, c:c + square] = 1
        #                 else:
        #                     break
        # for i in range(2, len(self.blocksNxN)):  # remove  3x3 blocks from 4x4 blocks and...
        #     self.blocksNxN[i - 1] -= self.blocksNxN[i]
        # print(self.blocksNxN)

        # self.state = (
        #     self.new_holes, self.new_pileHeight, self.altitudeDifference, self.empty,
        #     *self.blocksNxN[3:], *self.oneHotBoard)  # , )*self.oneHot
        # print(self.oneHotBoard)

        self.state = (np.reshape([self.oneHotBoard], (1, 24, 10, 1))).astype('float32')
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

        # self.reward = self.blocksNxN[2] * 1 + self.blocksNxN[3] * 2 + self.blocksNxN[4] * 2.5 \
        #               + self.blocksNxN[5] * 3 + self.blocksNxN[6] * 3.5 + self.blocksNxN[7] * 4 \
        #               + self.blocksNxN[8] * 4.5 + self.lowerBlocks
        # +(self.empty) * -0.125 + (self.pieces) * 1 + (self.new_holes - self.last_holes) * -.1 \
        # + (self.new_pileHeight - self.last_pileHeight) * -1 + self.rowFull * 100
        # self.reward = self.lowerBlocks * .000001
        self.reward = -(self.w1 * self.piece_holes + self.w2 * np.std(self.height_idx)) + \
                      (self.w3 * self.side + self.w4 * np.mean(self.piece_height))
        # *self.increase np.sum(self.piece_height)-np.var(self.height_idx)
        # print(np.mean(self.piece_height))

        # self.reward = self.side - np.var(self.height_idx) + self.pieces  # np.sum(self.piece_height) * self.pieces
        # self.reward = self.rowsFilled - self.lastRowsFilled
        return self.reward

    def setLabels(self):

        self.pieces_var.set('Pieces:{}'.format(self.pieces))
        self.holes_var.set('Holes:{}'.format(self.new_holes))
        self.empty_var.set('Empty:{}'.format(self.empty))
        # self.pileHeight_var.set('PileHeight: {}'.format(self.new_pileHeight))
        # self.altitudeDifference_var.set('Altitude Difference: {}'.format(self.altitudeDifference))
        # self.blocks_var.set('2x2 Blocks: {}\n3x3 Blocks: {}\n4x4 Blocks: {}\n5x5 Blocks: {}\n6x6 Blocks: {}'
        #                     '\n7x7 Blocks: {}\n8x8 Blocks: {}\n9x9 Blocks: {}'
        #                     .format(self.blocksNxN[2], self.blocksNxN[3], self.blocksNxN[4], self.blocksNxN[5],
        #                             self.blocksNxN[6], self.blocksNxN[7], self.blocksNxN[8], self.blocksNxN[9]))

    def __init__(self, parent, render):
        parent.title('RL Storage')
        self.parent = parent
        self.render = render  # show game board

        self.board_width = 10
        self.board_height = 24
        self.width = 300
        self.height = 720
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

        # for key in ('<Down>', '<Left>', '<Right>', 'a', 'A', 's', 'S', 'd', 'D'):
        #     self.parent.bind(key, self.shift)
        #
        # for key in ('0', 'q', 'Q', 'e', 'E', '<Up>', 'w', 'W'):
        #     self.parent.bind(key, self.rotate)
        #
        # self.parent.bind('<space>', self.snap)
        # self.parent.bind('<Escape>', self.pause)
        # self.parent.bind('n', self.reset)
        # self.parent.bind('N', self.reset)
        # self.parent.bind('g', self.toggle_guides)
        # self.parent.bind('G', self.toggle_guides)

        if self.render:
            self.canvas = None
            self.preview_canvas = None
            self.ticking = None
            self.spawning = None
            self.guide_fill = ''

            self.pieces_var = tk.StringVar()
            self.holes_var = tk.StringVar()
            self.empty_var = tk.StringVar()
            # self.pileHeight_var = tk.StringVar()
            # self.altitudeDifference_var = tk.StringVar()
            # self.blocks_var = tk.StringVar()

            self.pieces_label = tk.Label(parent, textvariable=self.pieces_var, width=20, height=1, font=('Tahoma', 12))
            self.pieces_label.grid(row=1, column=1, sticky='N')  # sticky: N S E W

            self.holes_label = tk.Label(parent, textvariable=self.holes_var, width=20, height=1, font=('Tahoma', 12))
            self.holes_label.grid(row=2, column=1, sticky='N')  # sticky: N S E W

            self.empty_label = tk.Label(parent, textvariable=self.empty_var, width=20, height=1, font=('Tahoma', 12))
            self.empty_label.grid(row=3, column=1, sticky='N')  # sticky: N S E W

            # self.pileHeight_label = tk.Label(parent, textvariable=self.pileHeight_var, width=20, height=1,
            #                                  font=('Tahoma', 12))
            # self.pileHeight_label.grid(row=4, column=1, sticky='N')
            #
            # self.altitudeDifference_label = tk.Label(parent, textvariable=self.altitudeDifference_var, width=20,
            #                                          height=1,
            #                                          font=('Tahoma', 12))
            # self.altitudeDifference_label.grid(row=5, column=1, sticky='N')
            #
            # self.blocks_label = tk.Label(parent, textvariable=self.blocks_var, width=20, height=10,
            #                              font=('Tahoma', 12))
            # self.blocks_label.grid(row=6, column=1, sticky='N')

    def toggle_guides(self, event=None):
        self.guide_fill = '' if self.guide_fill else 'black'
        self.canvas.itemconfig(self.guides[0], fill=self.guide_fill)
        self.canvas.itemconfig(self.guides[1], fill=self.guide_fill)

    def draw_board(self):
        # if self.spawning:
        #     self.parent.after_cancel(self.spawning)
        self.board = [['' for column in range(self.board_width)] for row in range(self.board_height)]

        if self.render:
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

            # if self.preview_canvas:
            #     self.preview_canvas.destroy()
            # self.preview_canvas = tk.Canvas(self.parent, width=5 * self.square_width, height=5 * self.square_width)
            # self.preview_canvas.grid(row=0, column=1)

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
                    if self.render:
                        square_idx = next(square_idxs)
                        coord = (column * self.square_width, row * self.square_width,
                                 (column + 1) * self.square_width, (row + 1) * self.square_width)
                        self.activePiece.coords[square_idx] = coord
                        self.canvas.coords(self.activePiece.piece[square_idx], coord)

        self.activePiece.row = r
        self.activePiece.column = c
        self.activePiece.shape = shape
        # self.move_guides(c, c + w)
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
        self.piece_height.clear()
        self.side = 0
        self.pieceIsActive = False
        for row in range(24):
            for col in range(10):
                if self.board[row][col] == '*':
                    self.side += self.board[row].count('x')
                    self.piece_height.append(row)
                    self.board[row][col] = 'x'

            # row[:] = ['x' if cell == '*' else cell for cell in row]
        if self.render:
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

    def spawn(self):
        key = random.choice('szrLoIT')
        # key = 'o'
        rot = random.choice((0, 90, 180, 270))
        # self.oneHotEncoder(key, rot)
        # key = 'o'
        # rot = 0
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

            if (self.render):
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

        # self.move_guides(start, (start + width))

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
