# default modules
import numpy as np
import os
from random import choice

from typing import NewType, Iterable, Literal

# bots
from requiem_bot.bot import *

# js some boilerplate imports!!

try:
    # installed modules
    import tkinter as tk

    from PIL import Image, ImageTk

except ModuleNotFoundError:
    print("Installing required modules from requirements.txt (sibling file). Please wait...")
    requirements = os.path.dirname(__file__) + "/requirements.txt"
    os.system("pip install -r \"" + requirements + "\"")
    print("Modules installed. Please restart the script.")
    exit()


####################################################################################################


# globals and pre-main functions go here :)


###################################################################################################


class Board:
    """The board class."""

    FENType = NewType('FENType', str)
    SAN = NewType('SAN', str)
    Coordinate = NewType('Coordinate', tuple[int, int])
    Move = NewType('Move', tuple[tuple[int, int], tuple[int, int]])

    WHITE = list(range(7,13))
    BLACK = list(range(1,7))

    piecetotoken = { # uppercase: white, lowercase: black
        " " : 0,
        "p" : 1,
        "n" : 2,
        "b" : 3,
        "r" : 4,
        "q" : 5,
        "k" : 6,
        "P" : 7,
        "N" : 8,
        "B" : 9,
        "R" : 10,
        "Q" : 11,
        "K" : 12
    }
    tokentopiece = ''.join(piecetotoken.keys())
    piecetosymbol = {
        " " : " ",
        "P" : "♟",
        "N" : "♞",
        "B" : "♝",
        "R" : "♜",
        "Q" : "♛",
        "K" : "♚",
        "p" : "♙",
        "n" : "♘",
        "b" : "♗",
        "r" : "♖",
        "q" : "♕",
        "k" : "♔",
    }

    singlecolmovedirs = {
        6 : [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        5 : [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        4 : [(-1, 0), (0, -1), (0, 1), (1, 0)],
        3 : [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        2 : [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)],
        1 : [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (0, 2), (0, -1), (0, -2)]
    }
    movedirs = {}

    # opposite colours
    for k, v in singlecolmovedirs.items():
        movedirs[k + 6] = v
        movedirs[k] = v
    
    can_move_long = [3,4,5,9,10,11]

    startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    ALL_TOKENS = [4,2,3,5,6,3,2,4,
                  1,1,1,1,1,1,1,1,
                  7,7,7,7,7,7,7,7,
                  10,8,9,11,12,9,8,10]

    # player 1: white, player 0: black

    def __init__(self, x: int = 8, y: int = 8, board = None) -> None:
        """The board constructor."""

        self.x = x
        self.y = y

        if board is not None:
            self.board = board
        else:
            self.board = self.FENtoboard(Board.startFEN)

        self.last_move: tuple[Board.Coordinate, Board.Coordinate, int] = () # start, end, taken piece

        self.tokens_on_board = [4,2,3,5,6,3,2,4,
                                1,1,1,1,1,1,1,1,
                                7,7,7,7,7,7,7,7,
                                10,8,9,11,12,9,8,10]

    def __str__(self) -> str:
        """Return a string representation of the board."""

        s = '+---' * self.x + '+\n'

        for y in range(self.y -1, -1, -1):
            for x in range(self.x):
                piece = self.tokentopiece[self.board[x][y]]
                s += f"| {self.piecetosymbol[piece]} "
            s += f'| {y + 1}\n'
            s += '+---' * self.x + '+\n'
        
        s += ''.join(f"  {chr(i + 96)} " for i in range(1, self.x + 1)) + '\n'

        return s

    def onboard(self, piece: int) -> bool:
        """Check if the piece is on the board."""

        for x in range(self.x):
            for y in range(self.y):
                if self.board[x][y] == piece:
                    return True
        return False

    def __getitem__(self, pos: Coordinate) -> int:
        """Return the counter at a position."""
        x, y = pos
        return self.board[x - 1][y - 1]

    def move(self, board: np.ndarray, start: Coordinate, end: Coordinate, promotion: int | None) -> np.ndarray:
        """Make a move and return the new board."""

        self.last_move = (start, end, board[end[0] - 1][end[1] - 1])
        
        if promotion is not None:
            board[end[0] - 1][end[1] - 1] = promotion
        else:
            board[end[0] - 1][end[1] - 1] = board[start[0] - 1][start[1] - 1]

        board[start[0] - 1][start[1] - 1] = 0

        return board
    
    def undo(self, board: np.ndarray) -> np.ndarray:
        """Undo a move and return the new board."""

        start, end, taken_piece = self.last_move

        board[start[0] - 1][start[1] - 1] = board[end[0] - 1][end[1] - 1]
        board[end[0] - 1][end[1] - 1] = taken_piece
        self.last_move = ()

        return board

    def boardtoFEN(self) -> FENType:
        """Convert the board to FEN."""

        FEN = ""

        for i in range(8):
            count = 0
            for j in range(8):
                if self.board[i][j] == 0:
                    count += 1
                else:
                    if count > 0:
                        FEN += str(count)
                        count = 0
                    FEN += self.tokentopiece[self.board[i][j]]
            if count > 0:
                FEN += str(count)
            FEN += "/"

        FEN = FEN[:-1]
        return FEN
    
    def FENtoboard(self, FEN: FENType) -> np.ndarray:
        """Convert the FEN to board."""

        self.board = np.zeros((self.x, self.y), dtype=int)
        FEN_index = 0
        x = 0
        y = 0
        
        while True:
            cell = FEN[FEN_index]
            FEN_index += 1

            if cell == " ":
                return self.board
            elif cell == "/":
                y += 1
                x = 0
                continue
            elif cell.isdigit():
                x += int(cell)
                continue
            else:
                self.board[x][self.y-y-1] = self.piecetotoken[cell]
                x += 1

            if FEN_index == len(FEN):
                return self.board
    
    def move_to_SAN(self, start: Coordinate, end: Coordinate, promotion: int | None) -> SAN:
        """Convert a move to SAN notation."""

        x2, y2 = end

        token = int(self[x2, y2])
        piece = self.tokentopiece[token]
        player = 1 if token in self.WHITE else 0
        has_taken = self.last_move[2] != 0

        self.undo(self.board)
        is_check = self.can_king_be_taken(1 - player, start, end, promotion)
        self.move(self.board, start, end, promotion)

        #########################

        if piece in "Pp" and has_taken:
            san = chr(start[0] + 96)
        elif piece in "Pp" and not has_taken:
            san = ''
        else:
            san = piece.upper()
        if has_taken:
            san += "x"
        san += f"{chr(x2 + 96)}{y2}"
        if promotion is not None:
            san += f"={self.tokentopiece[promotion]}"
            san = san[1:] # remove the initial piece letter
        if is_check:
            san += "+"
            input("CHECK! Press enter to continue...")

        return san

    def islegalmovewithoutcheck(self, start: Coordinate, end: Coordinate) -> bool:
        """Check if the move is legal."""

        x1, y1 = start
        x2, y2 = end
        if x1 not in range(1, self.x + 1) or y1 not in range(1, self.y + 1) or x2 not in range(1, self.x + 1) or y2 not in range(1, self.y + 1):
            return False

        cell = int(self[x1, y1])
        target = int(self[x2, y2])
        relative = (x2 - x1, y2 - y1)

        # empty start square
        if cell == 0:
            return False
        
        moves = self.movedirs[int(cell)]

        if cell in [1, 7]: # pawn
            return self.islegalpawnmovewithoutcheck(start, end)
        
        if cell not in Board.can_move_long:
            if relative not in moves:
                return False

        if cell in Board.can_move_long:
            passed_check = False
            for dx, dy in moves:

                # check the coordinates along each line
                for i in range(max(self.x, self.y)):
                    linecoord: Board.Coordinate = (x1 + dx * i, y1 + dy * i) # linecoord denotes the current coordinate being checked

                    if linecoord[0] not in range(1, self.x + 1) or linecoord[1] not in range(1, self.y + 1): # if the coordinate is out of bounds
                        break
                    
                    linecell = self[linecoord[0], linecoord[1]] # get the cell on the line
                    if linecell != 0 and linecoord not in [(x1, y1), (x2, y2)]: # if the cell isn't empty and isn't the starting or ending cell
                        break

                    # if the relative coordinates are equal to the change in x and y, the coordinate is so far valid
                    if relative == (dx * i, dy * i):
                        passed_check = True
                        break
                if passed_check:
                    break

            if not passed_check:
                return False

        # same color capture
        if cell in Board.WHITE and target in Board.WHITE:
            return False
        if cell in Board.BLACK and target in Board.BLACK:
            return False
        
        return True

    def islegalpawnmovewithoutcheck(self, start: Coordinate, end: Coordinate) -> bool:
        """Check if the pawn move is legal."""

        # assume the start is on a pawn

        x1, y1 = start
        x2, y2 = end
        if x1 not in range(1, self.x + 1) or y1 not in range(1, self.y + 1) or x2 not in range(1, self.x + 1) or y2 not in range(1, self.y + 1):
            return False

        cell = int(self[x1, y1])
        target = int(self[x2, y2])
        dx, dy = (x2 - x1, y2 - y1)

        # forwards / backwards
        if dx == 0:
            if dy > 0 and cell == 1: # black pawn moving forward
                return False
            if dy < 0 and cell == 7: # white pawn moving backward
                return False
                    
            # two spaces move; can only move 2 spaces on the 2nd / 7th rank
            if abs(dy) == 2 and y1 not in [2, 7]:
                return False
            
            # obstruction check
            if target != 0:
                return False
            elif abs(dy) == 2 and self[x1, y1 + dy // 2] != 0:
                return False
        
        elif dx != 0 and dy != 0: # diagonal move
            if target == 0: # must capture a piece diagonally
                return False
            # same color capture
            if cell == 7 and target in Board.WHITE: # white pawn own capture
                return False
            if cell == 1 and target in Board.BLACK: # black pawn own capture
                return False
            
        return True

    def allmoveswithoutcheck(self, player: Literal[0, 1]) -> Iterable[Move]:
        """Get all possible moves without checking checks."""

        present_pieces = []
        target_range = Board.WHITE if player == 1 else Board.BLACK
        for i, x in enumerate(self.board, start=1):
            for j, y in enumerate(x, start=1):
                if y in target_range:
                    present_pieces.append((y,(i,j)))

        for piece, (x,y) in present_pieces:
            if piece in [1,7]: # pawn

                for dx, dy in self.movedirs[piece]:
                    if self.islegalpawnmovewithoutcheck((x, y), (x + dx, y + dy)):
                        if piece == 7 and y + dy == self.y: # white pawn promotion
                            yield (x, y), (x + dx, y + dy), 8
                            yield (x, y), (x + dx, y + dy), 9
                            yield (x, y), (x + dx, y + dy), 10
                            yield (x, y), (x + dx, y + dy), 11
                            continue
                        elif piece == 1 and y + dy == 1: # black pawn promotion
                            yield (x, y), (x + dx, y + dy), 2
                            yield (x, y), (x + dx, y + dy), 3
                            yield (x, y), (x + dx, y + dy), 4
                            yield (x, y), (x + dx, y + dy), 5
                            continue
                        else:
                            yield (x, y), (x + dx, y + dy), None
            
            else: # other pieces

                for dx, dy in self.movedirs[piece]:
                    if piece in Board.can_move_long:
                        for i in range(max(self.x, self.y)):
                            linecoord: Board.Coordinate = (x + dx * i, y + dy * i) # linecoord denotes the current coordinate being checked
                            if self.islegalmovewithoutcheck((x, y), linecoord):
                                yield (x, y), linecoord, None
                    else:
                        if self.islegalmovewithoutcheck((x, y), (x + dx, y + dy)):
                            yield (x, y), (x + dx, y + dy), None

    def can_king_be_taken(self, player: Literal[0, 1], start: Coordinate, end: Coordinate, promotion: int | None) -> bool:
        """Check if the player king can be taken after a move."""
        
        # print("Initial board:", self)

        # make the initial move and create a new board class with the starting board as the new board
        board = self.board.copy()
        self.move(board, start, end, promotion)
        cb = Board(board=board)

        for start, end, promotion in cb.allmoveswithoutcheck(player):
            cb.move(cb.board, start, end, promotion)
            if player == 1 and not cb.onboard(6):
                return True
            elif player == 0 and not cb.onboard(12):
                return True
            cb.undo(cb.board)

        return False

    def islegalmove(self, start: Coordinate, end: Coordinate, promotion: int | None) -> bool:
        """Check if the move is legal."""

        player = 0 if self.board[start[0] - 1][start[1] - 1] in Board.WHITE else 1
        
        if self.can_king_be_taken(player, start, end, promotion):
            return False
        return True

    def force_update_tokens_on_board(self) -> None:
        """Forcibly update the tokens on the board."""
        self.tokens_on_board = self.board.flatten().tolist()

    def legal_moves(self, player: Literal[0, 1]) -> Iterable[Move] | False:
        """Get all legal moves."""

        legal_move_found = False
        
        for start, end, promotion in self.allmoveswithoutcheck(player):
            if self.islegalmove(start, end, promotion):
                # print(start, end, promotion)
                # if promotion is not None:
                #     input()
                legal_move_found = True
                yield start, end, promotion
        
        if not legal_move_found:
            return False
    
    def is_stalemate(self, player: Literal[0, 1]) -> bool:
        """Determine whether the board is stalemate."""

        # are there legal moves?
        if not self.legal_moves(player):
            print("No legal moves found.",bc='^')
            return True
        
        # insufficient material
        if any(token in self.tokens_on_board for token in [1,4,5,7,10,11]): # major pieces and pawns
            print("Major pieces and pawns found.",bc='^')
            return False
        
        if all(token in self.tokens_on_board for token in [2,3]): # black knight and bishop
            print("Black knight and bishop found.",bc='^')
            return False
        if all(token in self.tokens_on_board for token in [8,9]): # white knight and bishop
            print("White knight and bishop found.",bc='^')
            return False
        
        if any(self.tokens_on_board.count(token) > 1 for token in [2,3]): # white knight and bishop
            print("At least 2 white knights or bishops found.",bc='^')
            return False
        if any(self.tokens_on_board.count(token) > 1 for token in [8,9]): # black knight and bishop
            print("At least 2 black knights or bishops found.",bc='^')
            return False
        
        print("Insufficient material.",bc='^')
        return True


###################################################################################################


class Sprites:

    """Sprites class."""

    POSITIONS = {
        "P" : ((0,0), (32,32)),
        "B" : ((32,0), (64,32)),
        "N" : ((64,0), (96,32)),
        "R" : ((96,0), (128,32)),
        "Q" : ((128,0), (160,32)),
        "K" : ((160,0), (192,32)),

        "p" : ((0,32), (32,64)),
        "b" : ((32,32), (64,64)),
        "n" : ((64,32), (96,64)),
        "r" : ((96,32), (128,64)),
        "q" : ((128,32), (160,64)),
        "k" : ((160,32), (192,64)),
        
        "sP" : ((192,0), (208,16)),
        "sB" : ((208,0), (224,16)),
        "sN" : ((224,0), (240,16)),
        "sR" : ((240,0), (256,16)),
        "sQ" : ((256,0), (272,16)),

        "sp" : ((192,32), (208,48)),
        "sb" : ((208,32), (224,48)),
        "sn" : ((224,32), (240,48)),
        "sr" : ((240,32), (256,48)),
        "sq" : ((256,32), (272,48)),

        "board" : ((0, 64), (320, 384)),
        "availablemovesquare" : ((272, 0), (304, 32))
    }

    def __init__(self, path: str) -> None:
        """Initialize the sprites."""

        self.SCALE = 2

        self.spritesheet = Image.open(parent(__file__) + '/' + path)
        self.spritesheet = self.spritesheet.resize((self.spritesheet.width * self.SCALE, self.spritesheet.height * self.SCALE)) # scale

        self.sprites: dict[str, Image.Image] = {}
        self.load()

    def load(self) -> None:
        """Load the sprites."""

        for name, ((x1, y1), (x2, y2)) in Sprites.POSITIONS.items():
            self.sprites[name] = self.spritesheet.crop((x1 * self.SCALE, y1 * self.SCALE, x2 * self.SCALE, y2 * self.SCALE))
    
    def place(self, cv: tk.Canvas, spritename: str, x: int, y: int) -> None:
        """Place a sprite on the board."""

        sprite = self.sprites[spritename]
        tk_sprite = ImageTk.PhotoImage(sprite)

        cv.create_image(x * self.SCALE, y * self.SCALE, image=tk_sprite, anchor="nw")

        if not hasattr(cv, 'images'):
            cv.images = []  # create if it doesn't exist
        cv.images.append(tk_sprite)


###################################################################################################


class App(tk.Tk):

    """Main application class."""

    CAPTURED_PIECE_POS = {
        1 : [(32, 16), (48, 16), (64, 16), (80, 16), (96, 16), (112, 16), (128, 16), (144, 16)],
        2 : [(160, 16), (176, 16)],
        3 : [(192, 16), (208, 16)],
        4 : [(224, 16), (240, 16)],
        5 : [(256, 16)],

        7 : [(32, 0), (48, 0), (64, 0), (80, 0), (96, 0), (112, 0), (128, 0), (144, 0)],
        8 : [(160, 0), (176, 0)],
        9 : [(192, 0), (208, 0)],
        10 : [(224, 0), (240, 0)],
        11 : [(256, 0)]
    }

    def __init__(self) -> None:
        """Initialize the application."""

        super().__init__()

        self.title('Chess')

        self.cv = tk.Canvas(self, width=640, height=640, bg="white")
        self.cv.pack()
        
        self.s = Sprites("spritesheet.png")
        self.sprites = self.s.sprites

    def load_board(self, board: Board) -> None:
        """Load the board."""

        self.cv.images = [] # clear the board

        self.s.place(self.cv, "board", 0, 0)

        # draw the pieces
        for x in range(board.x):
            for y in range(board.y):
                token = int(board.board[x][board.y - 1 - y])
                if token != 0:
                    piece = board.tokentopiece[token]
                    self.s.place(self.cv, piece, x * 32 + 32, y * 32 + 32)

        taken = []

        for i in Board.piecetotoken.values():
            count = list(board.board.flatten()).count(i)
            taken.extend([i] * (Board.ALL_TOKENS.count(i) - count))

        captured_pos = {k : v.copy() for k, v in App.CAPTURED_PIECE_POS.items()}

        # draw the captured pieces
        for token in taken:
            captured_positions = captured_pos[token]
            x, y = captured_positions.pop(0)
            captured_pos[token] = captured_positions
            self.s.place(self.cv, 's' + board.tokentopiece[token], x, y)

        self.cv.update()


###################################################################################################


def play(board: Board, app: App, player: int) -> None:
    """Play the game."""
    
    legals = board.legal_moves(player)
    
    try:
        random_move = choice(list(legals))
    except IndexError:
        if board.can_king_be_taken(player, (0,0), (0,0), None):
            print("Checkmate!")
        else:
            print("Stalemate!")
        return
    
    if board.is_stalemate(player):
        print("Stalemate!")
        return

    board.board = board.move(board.board, random_move[0], random_move[1], random_move[2])
    board.force_update_tokens_on_board()

    app.load_board(board)
    print(f"Player {player} played {board.move_to_SAN(*random_move)}.")
    # input()
    
    app.after(1, play, board, app, 1 - player)


###################################################################################################


def main() -> None:
    """The main program."""

    app = App()
    board = Board()

    play(board, app, 1)

    app.mainloop()

if __name__ == '__main__':
    main()