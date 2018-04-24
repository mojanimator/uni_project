import tkinter as tk
import random

# class for music
try:
    import pygame as pg
except ImportError:
    audio = None
else:
    audio = True
from matrix_rotation import rotate_array as rotate


class Shape:
    def __init__(self, shape, key, piece, row, column, coords):
        self.shape = shape
        self.key = key
        self.piece = piece
        self.row = row
        self.column = column
        self.coords = coords


class Tetris:
    def __init__(self, parent, audio):
        parent.title('Tetris')
        self.parent = parent
        self.audio = audio
        if self.audio:
            pg.mixer.init(buffer=522)
            try:
                self.sounds = {name: pg.mixer.Sound(name) for name in
                               ('music.ogg', 'settle.ogg', 'clear.ogg', 'lose.ogg')}
            except pg.error as e:
                self.audio = None
                print(e)
            else:
                self.audio = {'m': True, 'f': True}
                for char in 'mMfF':
                    self.parent.bind(char, self.toggle_audio)
                self.sounds['music.ogg'].play(loops=-1)

        self.board_width = 10
        self.board_height = 24

        self.width = 300
        self.height = 720
        self.high_score = 0
        self.max_speed_score = 1000
        self.speed_factor = 30
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
        self.parent.bind('n', self.new)
        self.parent.bind('N', self.new)
        self.parent.bind('g', self.toggle_guides)
        self.parent.bind('G', self.toggle_guides)

        self.canvas = None
        self.preview_canvas = None
        self.ticking = None
        self.spawning = None
        self.guide_fill = ''
        self.score_var = tk.StringVar()
        self.score_var.set('Score:\n0')
        self.high_score_var = tk.StringVar()
        self.high_score_var.set('High Score:\n0')
        self.score_label = tk.Label(root, textvariable=self.score_var, width=20, height=5, font=('Arial Black', 12))
        self.score_label.grid(row=1, column=1, sticky='N')

        self.high_score_label = tk.Label(root, textvariable=self.high_score_var, width=20, height=5,
                                         font=('Arial Black', 12))
        self.high_score_label.grid(row=2, column=1)
        self.draw_board()

    def toggle_guides(self, event=None):
        self.guide_fill = '' if self.guide_fill else 'black'
        self.canvas.itemconfig(self.guides[0], fill=self.guide_fill)
        self.canvas.itemconfig(self.guides[1], fill=self.guide_fill)

    def toggle_audio(self, event=None):
        if not event: return
        key = event.keysym.lower()
        self.audio[key] = not self.audio[key]
        if key == 'm':
            if not self.audio['m']:
                self.sounds['music.ogg'].stop()
            else:
                self.sounds['music.ogg'].play(loops=-1)

    def draw_board(self):
        if self.ticking:
            self.parent.after_cancel(self.ticking)
        if self.spawning:
            self.parent.after_cancel(self.spawning)
        self.board = [['' for column in range(self.board_width)] for row in range(self.board_height)]
        self.field = [[None for column in range(self.board_width)] for row in range(self.board_height)]
        if self.canvas:
            self.canvas.destroy()
        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0, rowspan=4)
        self.h_separator = self.canvas.create_line(0, self.height // 6, self.width, self.height // 6, width=2)
        self.v_separator = self.canvas.create_line(self.width, 0, self.width, self.height, width=2)
        for c in range(self.board_width):
            self.canvas.create_line(c * self.square_width, 0, c * self.square_width, self.height, fill='lightgray')
        for r in range(self.board_height):
            self.canvas.create_line(0, r * self.square_width, self.width, r * self.square_width, fill='lightgray')
        if self.preview_canvas:
            self.preview_canvas.destroy()
        self.preview_canvas = tk.Canvas(root, width=5 * self.square_width, height=5 * self.square_width)
        self.preview_canvas.grid(row=0, column=1)

        self.tickrate = 1000
        self.score_var.set('Score:\n0')
        self.score = 0
        self.pieceIsActive = False

        self.paused = False
        self.preview()
        self.guides = [
            self.canvas.create_line(0, 0,
                                    0, self.height),
            self.canvas.create_line(self.width, 0,
                                    self.width, self.height)]
        self.spawning = self.parent.after(self.tickrate, self.spawn)
        self.ticking = self.parent.after(self.tickrate * 2, self.tick)

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

    def tick(self):
        if self.pieceIsActive:
            self.shift()
        self.ticking = self.parent.after(self.tickrate, self.tick)

        # self.parent.after(self.tickrate, self.tick)

    def rotate(self, event=None):
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
        direction = event.keysym
        if direction in {'q', 'Q'}:
            shape = rotate(self.activePiece.shape, -90)
            rotation_index = (self.activePiece.rotation_index - 1) % 4
            ra, rb = self.activePiece.rotation[rotation_index]
            rotation_offsets = -ra, -rb
        elif direction in {'e', 'E', '0', 'Up', 'w', 'W'}:
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

    def shift(self, event=None):
        down = {'Down', 's', 'S'}
        left = {'Left', 'a', 'A'}
        right = {'Right', 'd', 'D'}
        if not self.pieceIsActive:
            return
        r = self.activePiece.row
        c = self.activePiece.column
        l = len(self.activePiece.shape)
        w = len(self.activePiece.shape[0])
        direction = (event and event.keysym) or 'Down'
        if direction in down:
            rt = r + 1  # row temp
            ct = c  # column temp

        elif direction in left:
            rt = r
            ct = c - 1
        elif direction in right:
            rt = r
            ct = c + 1
        success = self.check_and_move(self.activePiece.shape, rt, ct, l, w)

        if direction in down and not success:
            self.settle()

    def settle(self):

        self.pieceIsActive = False
        for row in self.board:
            row[:] = ['x' if cell == '*' else cell for cell in row]

        for (x1, y1, x2, y2), id in zip(self.activePiece.coords, self.activePiece.piece):
            self.field[y1 // self.square_width][x1 // self.square_width] = id

        indices = [idx for idx, row in enumerate(self.board) if all(row)]
        if (indices):
            self.score += (1, 2, 5, 10)[len(indices) - 1]
            self.score_var.set('Score:\n{}'.format(self.score))
            self.clear(indices)
            if self.score <= self.max_speed_score:
                self.tickrate = 1000 // (self.score // self.speed_factor + 1)
        if any(any(row) for row in self.board[:4]):
            self.lose()
            return
        if self.audio['f'] and not indices:
            self.sounds['settle.ogg'].play()
        self.spawning = self.parent.after(self.tickrate, self.spawn)

    def preview(self):
        self.preview_canvas.delete(tk.ALL)
        key = random.choice('szrLoIT')
        shape = rotate(self.shapes[key], random.choice((0, 90, 180, 270)))
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

                    self.previewPiece.piece.append(
                        self.preview_canvas.create_rectangle(
                            self.previewPiece.coords[-1], fill=self.colors[key], width=2,
                            # outline='dark' + self.colors[key]
                        )
                    )
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
        self.pieceIsActive = True
        self.activePiece = self.previewPiece
        self.preview()
        width = len(self.activePiece.shape[0])
        start = (10 - width) // 2
        self.activePiece.column = start
        self.activePiece.start = start
        self.activePiece.coords = []
        self.activePiece.piece = []
        for y, row in enumerate(self.activePiece.shape):
            self.board[y][start:start + width] = self.activePiece.shape[y]
            for x, cell in enumerate(row, start=start):
                if cell:
                    self.activePiece.coords.append(
                        (
                            self.square_width * x,
                            self.square_width * y,
                            self.square_width * (x + 1),
                            self.square_width * (y + 1)
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
        # self.print_board()

    def new(self, event=None):
        self.draw_board()

    def lose(self):
        self.pieceIsActive = False
        if self.audio['f']:
            self.sounds['lose.ogg'].play()
        self.high_score = max(self.score, self.high_score)
        self.high_score_var.set('High Score:\n{}'.format(self.high_score))
        self.parent.after_cancel(self.ticking)
        self.parent.after_cancel(self.spawning)
        self.clear_iter(range(len(self.board)))
        self.new()

    def snap(self, event=None):
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
        if self.audio['f']:
            self.sounds['clear.ogg'].play()
        for idx in indices:
            self.board.pop(idx)
            self.board.insert(0, ['' for column in range(self.board_width)])
        self.clear_iter(indices)

    def clear_iter(self, indices, current_column=0):
        for row in indices:
            if row % 2:
                cc = current_column
            else:
                cc = self.board_width - current_column - 1

            id = self.field[row][cc]
            self.field[row][cc] = None
            self.canvas.delete(id)
        if current_column < self.board_width - 1:
            self.parent.after(50, self.clear_iter, indices, current_column + 1)
        else:
            for idx, row in enumerate(self.field):
                offset = sum(r > idx for r in indices) * self.square_width
                for square in row:
                    if square:
                        self.canvas.move(square, 0, offset)

            for row in indices:
                self.field.pop(row)
                self.field.insert(0, [None for x in range(self.board_width)])


root = tk.Tk()
tetris = Tetris(root, audio)
root.mainloop()
