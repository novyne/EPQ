# default modules
import numpy as np
import os
import sys

from random import choice
from typing import NewType, Literal, Iterable

# bots
from requiem_bot.bot import *

# js some boilerplate imports!!
sys.path.insert(0, "/Users/User/Desktop/vscode/python")
from imports import *

try:
    # installed modules
    open(parent(__file__) + '/requirements.txt', 'w').close()

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

    WHITE = list(range(1,7))
    BLACK = list(range(7,13))

    piecetotoken = {
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

    def __init__(self, x: int = 8, y: int = 8) -> None:
        """The board constructor."""

        self.x = x
        self.y = y

        self.board = self.FENtoboard(Board.startFEN)

        self.last_move: tuple[Board.Coordinate, Board.Coordinate, int] = () # start, end, taken piece

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

    def onboard(self, board: np.ndarray, piece: int) -> bool:
        """Check if the piece is on the board."""

        for x in range(self.x):
            for y in range(self.y):
                if board[x][y] == piece:
                    return True
        return False

    def __getitem__(self, pos: Coordinate) -> int:
        """Return the counter at a position."""
        x, y = pos
        return self.board[x - 1][y - 1]

    def move(self, board: np.ndarray, start: Coordinate, end: Coordinate) -> np.ndarray:
        """Make a move and return the new board."""

        self.last_move = (start, end, board[end[0] - 1][end[1] - 1])
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
    
    def move_to_SAN(self, start: Coordinate, end: Coordinate) -> SAN:
        """Convert a move to SAN notation."""

        x2, y2 = end

        token = int(self[x2, y2])
        piece = self.tokentopiece[token]
        has_taken = self.last_move[2] != 0

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

        # illegal move
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
            if cell == 1 and target in Board.WHITE: # white pawn own capture
                return False
            if cell == 7 and target in Board.BLACK: # black pawn own capture
                return False
            
        return True

    def allmoveswithoutcheck(self, player: Literal[0, 1]) -> Iterable[Move]:
        """Get all possible moves without checking checks."""

        # 0 : white, 1 : black
        if player not in [0, 1]:
            raise ValueError(f"Invalid player: {player}")

        present_pieces = []
        target_range = range(1, 7) if player == 0 else range(7, 13)
        for i, x in enumerate(self.board, start=1):
            for j, y in enumerate(x, start=1):
                if y in target_range:
                    present_pieces.append((y,(i,j)))

        for piece, (x,y) in present_pieces:
            if piece in [1,7]: # pawn
                for dx, dy in self.movedirs[piece]:
                    if self.islegalpawnmovewithoutcheck((x, y), (x + dx, y + dy)):
                        yield (x, y), (x + dx, y + dy)

            for dx, dy in self.movedirs[piece]:
                if piece in Board.can_move_long:
                    for i in range(max(self.x, self.y)):
                        linecoord: Board.Coordinate = (x + dx * i, y + dy * i) # linecoord denotes the current coordinate being checked
                        if self.islegalmovewithoutcheck((x, y), linecoord):
                            yield (x, y), linecoord
                else:
                    if self.islegalmovewithoutcheck((x, y), (x + dx, y + dy)):
                        yield (x, y), (x + dx, y + dy)

    def can_king_be_taken(self, player: Literal[0, 1]) -> bool:
        """Check if the player king can be taken."""

        board = self.board.copy()

        if player not in [0, 1]:
            raise ValueError(f"Invalid player: {player}")
            """Check if the king can be taken."""

        for start, end in self.allmoveswithoutcheck(player):
            self.move(board, start, end)
            if player == 0 and not self.onboard(board, 6):
                return True
            elif player == 1 and not self.onboard(board, 12):
                return True
            self.undo(board)

        return False

    def islegalmove(self, start: Coordinate, end: Coordinate) -> bool:
        """Check if the move is legal."""

        player = 0 if self.board[start[0] - 1][start[1] - 1] in Board.WHITE else 1

        board = self.board.copy()
        self.move(board, start, end)
        
        if self.can_king_be_taken(player):
            return False
        return True

    def legal_moves(self, player: Literal[0, 1]) -> Iterable[Move]:
        """Get all legal moves."""

        # 0 : white, 1 : black
        if player not in [0, 1]:
            raise ValueError(f"Invalid player: {player}")
        
        for start, end in self.allmoveswithoutcheck(player):
            if self.islegalmove(start, end):
                yield start, end


###################################################################################################


# functions go here :)


###################################################################################################


def main() -> None:
    """The main program."""

    board = Board()

    while True:
        for player in [0, 1]:
            legals = board.legal_moves(player)
            random_move = choice(list(legals))

            board.board = board.move(board.board, random_move[0], random_move[1])
            print(board)
            print(f"Player {player} played {board.move_to_SAN(*random_move)}.")
            input()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{mc}Goodbye!")