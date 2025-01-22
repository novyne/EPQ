# EPQ Menu
## Revision 1
### Creating of the Board

I began by using a pre-created template I use for the majority of my Python files.
```python
# default modules
import numpy as np
import os
from random import choice

from typing import NewType, Iterable, Literal

# bots
from requiem_bot.bot import *

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
```
Next, I created a class to hold the information of the chess board.
```python
class Board:
    """The board class."""

    FENType = NewType('FENType', str)
    SAN = NewType('SAN', str)
    Coordinate = NewType('Coordinate', tuple[int, int])
    Move = NewType('Move', tuple[tuple[int, int], tuple[int, int]])
```
I began by initialising some `NewType` instances which I could then use to *type-annotate* my functions to improve readability.
```python
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
```
Next, I represented each colour piece (and pawn) with 12 tokens, in the form of integers. I then created dictionaries (`piecetotoken` and `tokentopiece`) to interchange between string and integer representation for the pieces.

I then began focussing on means of representing the piece's movement. I eventually settled on a list of coordinates attached to each piece token dictating in which directions the piece could move.
```python
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
```

I used iteration to duplicate the movement for the black 
tokens as well. Additionally, I created a list containing 
piece tokens which had the ability to 'move long'; i.e. 
theoretically move infinitely assuming the size of the board
permitted it.

I then created **FEN** notation to represent the standard
starting position of a Chess board, as well as create a list
of tokens at the start of the game.

```python
startFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

ALL_TOKENS = [4,2,3,5,6,3,2,4,
                1,1,1,1,1,1,1,1,
                7,7,7,7,7,7,7,7,
                10,8,9,11,12,9,8,10]

# player 1: white, player 0: black
```

I then moved on to the `__init__` statement of the Board.

```python
def __init__(self, x: int = 8, y: int = 8, board = None) -> None:
        """The board constructor."""

        self.x = x
        self.y = y

        if board is not None:
            self.board = board
        else:
            self.board = self.FENtoboard(Board.startFEN)
```
I *globalised* the variables passed into the initialise
statement, as well as give the option to build a board
based off of an optional parameter.

For testing purposes, I wrote a function that returns a
text representation of the board when called.
```python
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
```

I continued with another *dunder* function to simplify
the retrieval of data from the board:

```python
def __getitem__(self, pos: Coordinate) -> int:
        """Return the counter at a position."""
        x, y = pos
        return self.board[x - 1][y - 1]
```

The following functions are self explanatory:

```python
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

        return san
```

Around this time, I began to realise my initial approach was
poor, and I created a new file to start anew, utilising
*bitboards* to better and faster represent the pieces.

## Revision 2
### Creating of the Board

The same boilerplate header:

```python
# default modules
import numpy as np
import os

from typing import Literal

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





###################################################################################################
```

#### Bitboards

I began by creating a Bitboard class, represented by a `numpy` *zeros* 8 by 8 array.
```python
class Bitboard:

    def __init__(self, bb: np.ndarray | None = None) -> None:
        """Bitboard class.
        Args:
            bb (np.ndarray | None, optional): The bitboard. Defaults to None.
            x (int, optional): The width of the bitboard. Defaults to 8.
            y (int, optional): The height of the bitboard. Defaults to 8.
            \nbb argument takes priority over x and y."""
        
        self.bb = np.zeros((8, 8), dtype=int) if bb is None else bb
```
I ensured to provide the option to change the X and Y dimensions of the bitboard in case I would need to use bitboards of different sizes.

What follows is a variety of *dunder* functions with examples for each function indicated by three right-facing arrows `>>>`. The bitboard is represented by `bb`.

```python
def __setitem__(self, key: tuple[int, int], value: int) -> None:
    self.bb[key] = value

>>> bb[2,3] = 1
```
The bit at index 2, 3 is now 1.
```python
def __getitem__(self, key: tuple[int, int]) -> int:
    return self.bb[key]

print(bb[2,3])
>>> Output: 1
```
The function has retrieved the bit at index 2, 3.
```python
def __add__(self, other: 'Bitboard') -> 'Bitboard':
    return Bitboard(self.bb + other.bb)
def __radd__(self, other: 'Bitboard') -> 'Bitboard':
    return Bitboard(self.bb + other.bb)

bb1 = Bitboard()
bb1[0,0] = 1
bb2 = Bitboard()
bb2[0,1] = 1
bb3 = bb1 + bb2
print(bb3[0,0], bb3[0,1])
>>> Output: 1 1
```
These functions act like `or` gates for bitboards. Any instance of 1 in either bitboard results in a 1 in the final bitboard.
```python
def __invert__(self) -> 'Bitboard':
    return Bitboard(~self.bb)
```
Reverses the polarity of each bit in the bitboard. At the time of writing, this function has no use.
```python
def __str__(self) -> str:
    return str(self.bb)
```
This function allows the bitboard to be printed.

Lastly, a non-dunder function completes the bitboard thus far.
```python
def on(self) -> bool:
    """Return whether the board contains any bits."""
    return np.any(self.bb)
```
This function is crucial to determine whether a certain piece type remains on the board. It is much faster and more concise than the previous approach:
```python
def onboard(self, piece: int) -> bool:
    """Check if the piece is on the board."""

    for x in range(self.x):
        for y in range(self.y):
            if self.board[x][y] == piece:
                return True
    return False
```
Rather than needing to iterate through the whole board, `numpy` provides a function that returns `True` if all values of the bitboard are not 0.

This is especially helpful to determine checks and checkmate; we can simply check if the respective coloured King bitboard is empty.

It also resolves issues where tokens would not properly be tracked and the game would think some pieces are still present.

#### The Pieces

Similar to my first approach, I needed a way to represent the pieces on the board before converting the board into bitboards.

My first approach was something like this;

```python
class Piece:

    def __init__(self, color: Literal['white', 'black'], movement: list[tuple[int,int]], movelong: bool) -> None:
        """Chess piece class.
        Args:
            color (Literal['white', 'black']): The color of the piece.
            movement (list[tuple[int,int]]): The short-hand movement of the piece.
            movelong (bool): Whether the piece can move 'long' or not.
        """

        global PIECE_ID_INDEX

        self.id = PIECE_ID_INDEX + 1
        PIECE_ID_INDEX += 1

        self.color = color
        self.movelong = movelong
        self.movement = self.expand_movement(movement)

        self.bb = Bitboard()
    
    def expand_movement(self, short_movement: list[tuple[int,int]]) -> list[tuple[int,int]]:
        """Expand the short movement list for all cases of movement."""

        mvmt = []
        for dx, dy in short_movement:
            for _ in range(2):
                mvmt.append((dx, dy))
                mvmt.append((dx, -dy))
                mvmt.append((-dx, dy))
                mvmt.append((-dx, -dy))
                dy, dx = dx, dy
        
        if not self.movelong:
            return list(set(mvmt))
        
        extended = []
        for scalar in range(1, 9):
            for dx, dy in list(set(mvmt)):
                extended.append((dx * scalar, dy * scalar))
        return extended

    def copy(self, 
             color: Literal['white', 'black'] | None = None, 
             movement: list[tuple[int,int]] | None = None, 
             movelong: bool | None = None) -> 'Piece':
        return Piece(self.color if color is None else color,
                     self.movement if movement is None else movement,
                     self.movelong if movelong is None else movelong)
```

```python
class White:
            def __init__(self) -> None:
                self.pawn = Pawn('white')
                self.knight = Piece('white', [(2,1)], False)
                self.bishop = Piece('white', [(1,1)], True)
                self.rook = Piece('white', [(1,0)], True)
                self.queen = Piece('white', [(1,0), (1,1)], True)
                self.king = self.queen.copy(movelong=False)

                self.pieces = [self.pawn, self.knight, self.bishop, self.rook, self.queen, self.king]

            def print_ids(self) -> None:
                """Print the IDs of each piece."""

                names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
                for name, piece in zip(names, self.pieces):
                    print(f"white {name} : {piece.id}")
        
        self.white = w = White()

        class Black:
            def __init__(self) -> None:
                self.pawn = Pawn(color='black')
                self.knight = w.knight.copy(color='black')
                self.bishop = w.bishop.copy(color='black')
                self.rook = w.rook.copy(color='black')
                self.queen = w.queen.copy(color='black')
                self.king = w.king.copy(color='black')

                self.pieces = [self.pawn, self.knight, self.bishop, self.rook, self.queen, self.king]

            def print_ids(self) -> None:
                """Print the IDs of each piece."""

                names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
                for name, piece in zip(names, self.pieces):
                    print(f"black {name} : {piece.id}")
        
        self.black = Black()
```

However, I soon realised this approach was impractical. The two classes share a near-identical function `print_ids` and otherwise have very similar data.