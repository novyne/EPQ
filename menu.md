# EPQ Menu - The Board
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

I devised a new approach;
```python
PIECE_ID_INDEX = 0

class Piece:

    def __init__(self, color: Literal['white', 'black'], movement: list[tuple[int, int]], movelong: bool) -> None:
        """Chess piece class.
        Args:
            color (Literal['white', 'black']): The color of the piece.
            movement (list[tuple[int, int]]): The short-hand movement of the piece.
            movelong (bool): Whether the piece can move 'long' or not.
        """

        global PIECE_ID_INDEX
        self.id = PIECE_ID_INDEX + 1
        PIECE_ID_INDEX += 1

        self.color = color
        self.movelong = movelong
        self.movement = self.expand_movement(movement)
        self.bb = Bitboard()

    def expand_movement(self, short_movement: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Expand the short movement list for all cases of movement.
        Args:
            short_movement (list[tuple[int,int]]): The short-movement list.
        Returns:
            list[tuple[int,int]]: The expanded movement list.
        """

        mvmt = []
        for dx, dy in short_movement:
            for _ in range(2):
                mvmt.append((dx, dy))
                mvmt.append((dx, -dy))
                mvmt.append((-dx, dy))
                mvmt.append((-dx, -dy))
                dy, dx = dx, dy

        # if not 'long', return the unique short movements
        if not self.movelong:
            return list(set(mvmt))

        # if 'long', scale the movements
        extended = [(dx * scalar, dy * scalar) for scalar in range(1, 9) for dx, dy in list(set(mvmt))]
        return extended

    def copy(self, color: Literal['white', 'black'] = None, movement: list[tuple[int, int]] = None, movelong: bool = None) -> 'Piece':
        """Create a copy of the piece, with optional overrides."""

        return Piece(
            self.color if color is None else color,
            self.movement if movement is None else movement,
            self.movelong if movelong is None else movelong
        )

class Player:

    def __init__(self, color: Literal['white', 'black']) -> None:
        """Player class to manage pieces for a given colour.
        Args:
            color (Literal['white', 'black']): The colour of the piece.
        """

        self.color = color
        self.pieces = self.create_pieces(color)

    def create_pieces(self, color: Literal['white', 'black']) -> list[Piece]:
        """Create pieces based on the color of the player.
        Args:
            color (Literal['white', 'black']): The colour of the piece.
        Returns:
            list[Piece]: A list of the created pieces.
        """

        pawn = Pawn(color=color)
        knight = Piece(color=color, movement=[(2, 1)], movelong=False)
        bishop = Piece(color=color, movement=[(1, 1)], movelong=True)
        rook = Piece(color=color, movement=[(1, 0)], movelong=True)
        queen = Piece(color=color, movement=[(1, 0), (1, 1)], movelong=True)
        king = queen.copy(color=color, movelong=False)

        return [pawn, knight, bishop, rook, queen, king]

    def print_ids(self) -> None:
        """Print the IDs of each piece for this player."""
        names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        for name, piece in zip(names, self.pieces):
            print(f"{self.color} {name} : {piece.id}")


class Board:

    def __init__(self) -> None:

        self.white = Player('white')
        self.black = Player('black')
        self.board = np.zeros((8, 8), dtype=int)
```

Now the organisation is much more concise and I do not repeat myself.

However, I realised I don't need a separate function to define the pieces; in fact, it makes my situation harder since I would have to access the pieces by list. Instead, I defined them inside of `Player.__init__`:
```python
class Player:

    def __init__(self, color: Literal['white', 'black']) -> None:
        """Player class to manage pieces for a given colour.
        Args:
            color (Literal['white', 'black']): The colour of the piece.
        """

        self.color = color

        self.pawn = Pawn(color=color)
        self.knight = Piece(color=color, movement=[(2, 1)], movelong=False)
        self.bishop = Piece(color=color, movement=[(1, 1)], movelong=True)
        self.rook = Piece(color=color, movement=[(1, 0)], movelong=True)
        self.queen = Piece(color=color, movement=[(1, 0), (1, 1)], movelong=True)
        self.king = self.queen.copy(color=color, movelong=False)

        self.pieces = [self.pawn, self.knight, self.bishop, self.rook, self.queen, self.king]

    def print_ids(self) -> None:
        """Print the IDs of each piece for this player."""
        names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        for name, piece in zip(names, self.pieces):
            print(f"{self.color} {name} : {piece.id}")
```

I now needed a way to take the state of a board (presumably a numpy 8x8 array using the IDs of the pieces).

To avoid using long statements such as `self.black.rook.id` to collect the IDs, I defined a dictionary as such:

```python
self.pieces = self.white.pieces.copy() + self.black.pieces.copy()

self.id = {}
for piece in self.pieces:
    self.id[piece] = id
```

Just as I created it, I realised this approach is no better than the first approach. Instead, I would define the structure then iteratively convert each piece into its ID.

```python
def default(self) -> list[list[int]]:
    """Return the default Chess board in piece IDs."""

    black = self.black
    white = self.white

    struct: list[list[Piece | Pawn]] = [
        [white.rook, white.knight, white.bishop, white.queen, white.king, white.bishop, white.knight, white.rook],
        [white.pawn] * 8,
        [0] * 8,
        [0] * 8,
        [0] * 8,
        [0] * 8,
        [black.pawn] * 8,
        [black.rook, black.knight, black.bishop, black.queen, black.king, black.bishop, black.knight, black.rook],
        ]
    
    struct_ids: list[list[int]] = []
    for row in struct:
        row_ids: list[int] = []
        for piece in row:
            row_ids.append(piece.id)
        struct_ids.append(row_ids)
    
    return np.array(struct_ids)
```

I opted to return a Board instance instead, such that I could create the board in one statement.

```python
board = np.array(struct_ids)
return Board(board=board)
```
```python
board = Board().default()
```

Next, I needed to represent the board in a set of bitboards stored in each pieces' data, then safely forget about the board state.

I iterated through each cell in the board:

```python
def write_bitboards_from_board(self) -> None:
    """Write the piece bitboards from a board state."""

    for x in range(8):
        for y in range(8):
            cell = self.board[x, y]
```
Then created a dictionary to access pieces by their ID:
```python
id = {piece.id : piece for piece in self.pieces}
```
Lastly, change the bit to 1 where the ID accords to the piece:
```python
id[cell].bb[x, y] = 1
```

The complete function:
```python
def write_bitboards_from_board(self) -> None:
    """Write the piece bitboards from a board state."""

    id = {piece.id : piece for piece in self.pieces}

    for x in range(8):
        for y in range(8):
            cell = self.board[x, y]
            id[cell].bb[x, y] = 1
```

I appended the function just after the board definition.
```python
self.board = np.zeros((8, 8), dtype=int) if board is None else board
self.write_bitboards_from_board()
```

#### Troubleshooting

This error arose:
```
File "/workspaces/EPQ/mainbitboard.py", line 186, in write_bitboards_from_board
    id[cell].bb[x, y] = 1
    ~~^^^^^^
KeyError: np.int64(0)
```

I resolved it by ensuring the cell was converted to a normal integer before trying to access the ID.
```python
cell = int(self.board[x, y])
```
Another error:
```
  File "/workspaces/EPQ/mainbitboard.py", line 186, in write_bitboards_from_board
    id[cell].bb[x, y] = 1
    ~~^^^^^^
KeyError: 0
```

I forgot to account for the missing 0 ID indicating an empty space, so I added a statement to ignore if the ID is 0:
```python
cell = int(self.board[x, y])
if cell == 0: continue
id[cell].bb[x, y] = 1
```

The bitboard loading function worked correctly now, however:
```
File "/workspaces/EPQ/mainbitboard.py", line 291, in main
    board = Board().default()
            ^^^^^^^^^^^^^^^^^
  File "/workspaces/EPQ/mainbitboard.py", line 214, in default
    row_ids.append(piece.id)
                   ^^^^^^^^
AttributeError: 'int' object has no attribute 'id'
```
Initially, I was unsure why the `piece` variable was an integer, so I printed it before the error.

```
...
<__main__.Pawn object at 0x7a6fe30102f0>
<__main__.Pawn object at 0x7a6fe30102f0>
<__main__.Pawn object at 0x7a6fe30102f0>
<__main__.Pawn object at 0x7a6fe30102f0>
0
Traceback (most recent call last):
  File "/workspaces/EPQ/mainbitboard.py", line 301, in <module>
    main()
    ...
```

I then realised that the zeros in the structure were in fact integers, and I would need to account for this.
```python
for piece in row:
    if piece == 0:
        row_ids.append(0)
    else:
        row_ids.append(piece.id)
```

```
File "/workspaces/EPQ/mainbitboard.py", line 177, in __init__
    self.write_bitboards_from_board()
  File "/workspaces/EPQ/mainbitboard.py", line 188, in write_bitboards_from_board
    id[cell].bb[x, y] = 1
    ~~^^^^^^
KeyError: 4
```

I printed the dictionary as well as creating dunder `__str__` functions for the `Piece` and `Pawn` classes.
Piece:
```py
def __str__(self) -> str:
    return f"{self.color} piece ID{self.id}"
```
Pawn:
```py
def __str__(self) -> str:
    return f"{self.color} pawn ID{self.id}"
```
ID dictionary:
```py
print('\t'.join([str(k) + ' ' + str(v) for k, v in id.items()]))
```
Output:
```
13 white pawn ID13      14 white piece ID14     15 white piece ID15     16 white piece ID16     17 white piece ID17     18 white piece ID18     19 black pawn ID19      20 black piece ID20     21 black piece ID21       22 black piece ID22     23 black piece ID23     24 black piece ID24
```

The mistake was that I had neglected to reset the `PIECE_ID_INDEX` counter after the creation of the board. Since the board was being called as `Board().default()`, 2 board states were being corrected and I would have to reset the counter between the two.

I inserted this at the top of the `Board.__init__` function:
```py
global PIECE_ID_INDEX
PIECE_ID_INDEX = 0
```

No errors arose this time, but I would have to check the bitboards to ensure their data was loaded correctly.
```
ID: 1
[[0 0 0 0 0 0 0 0]
 [1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]]
ID: 2
[[0 1 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]

 ...

 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0]]
ID: 12
[[0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0]]
```

The bitboards are working correctly. For example, ID `1` indicates white pawns, and there is a distinct row of 1s shown in its bitboard.

The bitboards display in reverse; bottom to top, left to right. This is why the white pawns are at the top, but when displayed properly, they will be at the bottom.

### Game Logic

#### Legal Moves (without check)

The function began as so:
```python
def legal_nocheck(self) -> list[list[tuple[int,int], tuple[int,int]]]:
    """Get all legal moves without checking for checks.
    Returns:
        list[list[tuple[int,int], tuple[int,int]]]: A list of pairs of coordinates describing the movement."""
```

I then figured I would need a function to return pieces that are still present on the board:

```python
def present_pieces(self) -> list[Piece | Pawn]:
    """Get the present piece types; i.e. pieces that do not have an empty bitboard.
    Returns:
        list[Piece | Pawn]: The list of present pieces.
    """

    present = []
    for piece in self.pieces:
        if piece.bb.any(): # does the bitboard have any 1s?
            piece.append(present)
    return present
```

The beginnings of the function:
```py
def legal_nocheck(self) -> list[list[Coordinate, Coordinate]]:
    """Get all legal moves without checking for checks.
    Returns:
        list[list[Coordinate, Coordinate]]: A list of pairs of coordinates describing the movement.
    """

    present = self.present_pieces()

    for piece in present:
        if isinstance(piece, Pawn):
            pass
        elif isinstance(piece, Piece):
            for dx, dy in piece.movement():
                # ?
```

Here, I would need a means of finding the coordinates of all 1s in each bitboard. Using iteration would render the bitboard redundant, however, `numpy` has a function that does precisely what I need from it. I wrote a new `Bitboard` function:
```py
def pos(self) -> list[Coordinate]:
    """Return the indices of any 1s in the bitboard.
    Returns:
        list[Coordinate]: The list of indices pointing to 1s.
    """

    pos = np.nonzero(self.bb)
    return list(zip(pos[0], pos[1]))
```

I then updated the `legal_nocheck` function:
```python
present = self.present_pieces()

for piece in present:
    if isinstance(piece, Pawn):
        pass
    elif isinstance(piece, Piece):
        for x, y in piece.bb.pos():
            for dx, dy in piece.movement:
                rx, ry = x + dx, y + dy
```

`rx` and `ry` now pointed to the resultant coordinates of a piece's movement.

The final few checks involved off-board checks and obstruction checks, which is where I realised my earlier approach to determining movement was inefficient:
```py
"""In Piece.expand_movement:"""

# if 'long', scale the movements
extended = [(dx * scalar, dy * scalar) for scalar in range(1, 9) for dx, dy in list(set(mvmt))]
return extended
```

I realised it would be better to instead not scale the movement and scale it whilst checking for obstructions in the legal move checker. I removed the scaling section of `Piece.expand_movement`, and planned to readd it in `legal_nocheck`.

**24/01/25**

```py
def legal_nocheck(self, color: Literal['white', 'black']) -> list[list[Coordinate, Coordinate]]:
    """Get all legal moves without checking for checks.
    Returns:
        list[list[Coordinate, Coordinate]]: A list of pairs of coordinates describing the movement.
    """

    white_bb = Bitboard().sum([piece.bb for piece in self.white.pieces])
    black_bb = Bitboard().sum([piece.bb for piece in self.black.pieces])

    self_bb = white_bb if color == 'white' else black_bb
    other_bb = black_bb if color == 'white' else white_bb
```

I added a colour parameter; the colour depends on the legal moves, of course. White pieces cannot take white pieces, for example, and most importantly, only the white player can move white pieces.

Some reorganisation later:

```py
def legal_nocheck(self, color: Literal['white', 'black']) -> list[list[Coordinate, Coordinate]]:
    """Get all legal moves without checking for checks.
    Returns:
        list[list[Coordinate, Coordinate]]: A list of pairs of coordinates describing the movement.
    """

    if color not in ['white', 'black']:
        raise ValueError(f"Incorrect color specified (got {color}, only 'white' and 'black' permitted).")

    self_pieces = self.white.pieces if color == 'white' else self.black.pieces
    other_pieces = self.black.pieces if color == 'white' else self.white.pieces

    self_bb = Bitboard().sum([piece.bb for piece in self_pieces])
    other_bb = Bitboard().sum([piece.bb for piece in other_pieces])

    moveable_pieces = list(set(self.present_pieces()) & set(self_pieces))
    
    # No legals list: I decided to yield instead.
```

Starting on the logic:

```py
for piece in moveable_pieces:
    if isinstance(piece, Pawn):
        pass #TODO
    elif isinstance(piece, Piece):
        for x, y in piece.bb.pos():
            for dx, dy in piece.movement:
                rx, ry = x + dx, y + dy

                # cannot 'capture' own piece
                if (rx, ry) in self_bb_pos:
                    continue

                # cannot movelong
                if not piece.movelong:
                    yield [(x, y), (rx, ry)]
                    continue
```

Now, I needed to work on the scalar. I also ensured to add an out of bounds check for the moveshort pieces.
```py
# out of bounds
if rx not in range(8) or ry not in range(8):
    continue
```

For clarity, I moved some code to a new function:

```py
def piece_legal_nocheck(self, piece: Piece) -> Iterable[list[Coordinate, Coordinate]]:
    """Get all legal moves for a single piece, specified by class.
    Args:
        piece (Piece): The piece to check for legal moves.
    Returns:
        Iterable[list[Coordinate, Coordinate]]: An iterable of pairs of coordinates describing the movement."""

    self_pieces = self.white.pieces if piece.color == 'white' else self.black.pieces
    other_pieces = self.black.pieces if piece.color == 'white' else self.white.pieces

    self_bb = Bitboard().sum([piece.bb for piece in self_pieces])
    self_bb_pos = self_bb.pos()

    for x, y in piece.bb.pos():
        for dx, dy in piece.movement:
            rx, ry = x + dx, y + dy

            # cannot 'capture' own piece
            if (rx, ry) in self_bb_pos:
                continue
            # out of bounds
            if rx not in range(8) or ry not in range(8):
                continue

            # cannot movelong
            if not piece.movelong:
                yield [(x, y), (rx, ry)]
                continue
```

Onto the scalar section. I decided I would iterate with a scalar `s` from 1 to 8, and multiply the magnitude of the direction by `s`. If the path is unobstructed, I would yield the movement, otherwise stop the iteration and continue to the next movement direction.

```py

"""In piece_legal_nocheck:"""
for s in range(1, 9):
    # scaled direction and resultant X and Y
    sdx, sdy = dx * s, dy * s
    rx, ry = x + sdx, y + sdy

    # cannot be out of bounds
    if rx not in range(8) or ry not in range(8):
        break

    # cannot 'capture' own piece
    if (rx, ry) in self_bb_pos:
        break
    # if capturing enemy piece, yield then break
    if (rx, ry) in other_bb_pos:
        yield [(x, y), (rx, ry)]
        break

    # otherwise yield
    yield [(x, y), (rx, ry)]
```

To link it back to `legal_nocheck`:
```py
"""In legal_nocheck:"""
for move in self.piece_legal_nocheck(piece):
    yield move
```

#### Cleaning Up

**27/01/2025**

I wanted to take a brief step away from obtaining legal moves, and focus on some other sections that needed more attention.

Firstly, I changed the `Pawn` class to be derived from the `Piece` class for ease.

```py
class Pawn(Piece):

    def __init__(self, color: Literal['white', 'black']) -> None:
        super().__init__(color, [], False)
    
    def copy(self, color: Literal['white', 'black'] = None) -> 'Pawn':
        """Create a copy of the pawn, with optional overrides."""

        return Pawn(
            self.color if color is None else color
        )
```

Secondly, I wrote a movement function which would both update the board and the bitboards. This would require a dictionary linking a piece ID to the respective piece.

```py
self.id: dict[int, Piece | Pawn] = {}
for piece in self.pieces:
    self.id[piece.id] = piece
```

```py
def move(self, x1: int, y1: int, x2: int, y2: int) -> None:
    """Move a piece at (x1, x2) to (y1, y2).
    Args:
        x1, y1: The coordinates of the start.
        x2: y2: The coordinates of the end.
    """

    start_id = self.board[x1, y1]
    start_piece = self.id[start_id]
    end_id = self.board[x2, y2]
    end_piece = self.id[end_id]

    # on start bb: set to 0 at start and set to 1 at end
    start_piece.bb[x1, y1] = 0
    start_piece.bb[x2, y2] = 1
    # on end bb: set to 0 at end
    end_piece.bb[x2, y2] = 0
```

I debated on whether I should update the board as well. If I didn't need to, I could always remove it later.

```py
# update board
self.board[x1, y1] = 0
self.board[x2, y2] = start_id
```

I then decided to test the legal move function so far:
```py
legals = board.legal_nocheck('white')

for m1, m2 in legals:
    print(m1[0], m1[1],'to',m2[0], m2[1])
```
Output:
```
0 1 to 2 2
0 1 to 2 0
0 6 to 2 7
0 6 to 2 5
```

It worked just fine; only pieces (not pawns) are able to move, and since only knights can jump over pieces, they are the only ones that are capable of movement.

For the future, I formatted the coordinates into Chess coordinates:

```python
for m1, m2 in legals:
    print(f"{chr(m1[0] + 97)}{m1[1] + 1} -> {chr(m2[0] + 97)}{m2[1] + 1}")
```
Output:
```
a2 -> c3
a2 -> c1
a7 -> c8
a7 -> c6
```

**29/01/2025**

Minor refactor:

*Before*
```py
def present_pieces(self) -> list[Piece | Pawn]:
    """Get the present piece types; i.e. pieces that do not have an empty bitboard.
    Returns:
        list[Piece | Pawn]: The list of present pieces.
    """

    present: list[Piece | Pawn] = []
    for piece in self.pieces:
        if piece.bb.any(): # does the bitboard have any 1s?
            present.append(piece)
    return present
```
*After*
```py
def present_pieces(self) -> list[Piece | Pawn]:
    """Get the present piece types; i.e. pieces that do not have an empty bitboard."""
    return [piece for piece in self.pieces if piece.bb.any()]
```

#### Pawns

Pawns are arguably and unfortunately the most complex 'piece' in terms of the way it moves, hence why I designated it a separate class.

I first decided to write a function inside the `Pawn` class which takes the X and Y of the respective pawn.

However, the movement of a pawn is also heavily dependent on the board state, so I moved the function to the `Board` class.

```py
def get_pawn_movement(self, x: int, y: int) -> list[list[Coordinate, Coordinate]]:
    """Obtain the possible movement of a pawn based on its position and colour.
    Args:
        x (int): The X coordinate of the pawn.
        y (int): The Y coordinate of the pawn.
    """ 
```

It may also be easier to pass the Pawn instance to the function as well, meaning that it is easier to identify its colour.

It was around this point where I realised I was repeating the following phrase (or variants of such) frequently:
```py
self_pieces = self.white.pieces if pawn.color == 'white' else self.black.pieces
other_pieces = self.black.pieces if pawn.color == 'white' else self.white.pieces

self_bb = Bitboard().sum([piece.bb for piece in self_pieces])
self_bb_pos = self_bb.pos()
other_bb = Bitboard().sum([piece.bb for piece in other_pieces])
other_bb_pos = other_bb.pos()
```

Even if the phrase isn't particularly computationally expensive, it might be a good idea to store the data in the `Board` class and redefine it each time I need to obtain the legal moves.

```py
"""At the bottom of the Board.__init__ constructor:"""

self.self_pieces = self.white.pieces if self.turn.color == 'white' else self.black.pieces
self.other_pieces = self.black.pieces if self.turn.color == 'white' else self.white.pieces

self.self_bb = Bitboard().sum([piece.bb for piece in self.self_pieces])
self.self_bb_pos = self.self_bb.pos()
self.other_bb = Bitboard().sum([piece.bb for piece in self.other_pieces])
self.other_bb_pos = self.other_bb.pos()
```

I then went through and replaced all the necessary variable names.

Returning back to the function in question, I would first begin by checking if the space ahead of the pawn is available.

The direction I check in would depend on the colour of the pawn, so:

```py
"""In get_pawn_movement:"""

movement: list[list[Coordinate, Coordinate]] = []

# y direction to check in
if pawn.color == 'white':
    dy = 1
else:
    dy = -1

# check for the space ahead
if (x, y+dy) not in self.all_pos:
    movement.append([(x, y), (x, y + dy)])
```
Next, I would check for any captures possible based on the spaces diagonally in front of the pawn.
```py
# capture
for dx in (1, -1):
    if (x+dx, y+dy) in self.other_pos:
        movement.append([(x, y), (x+dx, y+dy)])
```
Finally, I would check whether the pawn can move two steps forward. I can save some computational effort by reorgnising the order of operations:

```py
# y direction to check in
if color == 'white':
    dy = 1
else:
    dy = -1
    
# capture
for dx in (1, -1):
    if (x+dx, y+dy) in self.other_pos:
        movement.append([(x, y), (x+dx, y+dy)])

# check for the space ahead
if (x, y+dy) not in self.all_pos:
    movement.append([(x, y), (x, y+dy)])
```

This way, if the space ahead is not free, I can skip the two-squares check.

```py
# check if the square ahead is occupied
if (x, y+dy) in self.all_pos:
    return movement
movement.append([(x, y), (x, y+dy)])

# check whether the pawn can move 2 squares forward
if (x, y+(dy*2)) in self.all_pos:
    return movement
if color == 'white' and y == 1:
    movement.append([(x, y), (x, y+(dy*2))])
elif color == 'black' and y == 6:
    movement.append([(x, y), (x, y+(dy*2))])
return movement
```

**Final function:**
```py
def get_pawn_movement(self, x: int, y: int, color: Literal['white','black']) -> list[list[Coordinate, Coordinate]]:
    """Obtain the possible movement of a pawn based on its position and colour.
    Args:
        x (int): The X coordinate of the pawn.
        y (int): The Y coordinate of the pawn.
    """

    movement: list[list[Coordinate, Coordinate]] = []

    # y direction to check in
    if color == 'white':
        dy = 1
    else:
        dy = -1

    # capture
    for dx in (1, -1):
        if (x+dx, y+dy) in self.other_pos:
            movement.append([(x, y), (x+dx, y+dy)])

    # check if the square ahead is occupied
    if (x, y+dy) in self.all_pos:
        return movement
    movement.append([(x, y), (x, y+dy)])

    # check whether the pawn can move 2 squares forward
    if (x, y+(dy*2)) in self.all_pos:
        return movement
    if color == 'white' and y == 1:
        movement.append([(x, y), (x, y+(dy*2))])
    elif color == 'black' and y == 6:
        movement.append([(x, y), (x, y+(dy*2))])
    return movement
```
The only thing to do now was to insert the movement into `legal_nocheck`.
```py
"""In legal_nocheck:"""

for piece in moveable_pieces:
    if isinstance(piece, Pawn):
        pass # <-- Here is where I need to insert the function.
    elif isinstance(piece, Piece):
        for move in self.piece_legal_nocheck(piece):
            yield move
```
Result:
```py
for piece in moveable_pieces:
    if isinstance(piece, Pawn):
        for x, y in piece.bb.pos():
            for move in self.get_pawn_movement(x, y, piece.color):
                yield move
    ...
```

For testing purposes, I commented out the piece movement and ran `legal_nocheck`.
Output:
```
b8 -> b9
```

This was highly indicative of an error - there is only one move recognised as legal. Furthermore, `b9` is not a valid Chess coordinate.

To start with, I printed out the x and y coordinate and the colour of each pawn as it was being tested.

Output:
```
1 0 white
1 1 white
1 2 white
1 3 white
1 4 white
1 5 white
1 6 white
1 7 white
b8 -> b9
```
This was very strange - the X and Y coordinates appear to have swapped places.
In an attempt to solve the issue, I reversed the coordinates when calling the function:
```py
for piece in moveable_pieces:
    if isinstance(piece, Pawn):
        for x, y in piece.bb.pos():
            for move in self.get_pawn_movement(y, x, piece.color):
                yield move
```
Output:
```
0 1 white
1 1 white
2 1 white
c2 -> c3
c2 -> c4
3 1 white
d2 -> d3
d2 -> d4
4 1 white
e2 -> e3
e2 -> e4
5 1 white
f2 -> g3
f2 -> f3
f2 -> f4
6 1 white
g2 -> h3
7 1 white
h2 -> g3
```
**30/01/25**
I figured this had something to do with a mishap with the bitboards, so I printed both bitboards to the console.

```py
"""In legal_nocheck:"""
print(self.self_bb)
print(self.other_bb)
print(self.turn.pawn.bb)
```
Output:
```
[[1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]]
[[0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1]]
[[0 0 0 0 0 0 0 0]
 [1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]]
```
Then, I realised the problem arose from the retrieval of the positions of pieces and pawns. The X and Y axes were flipped, so in `Bitboard.pos()`, I reversed the coordinates outputted and changed the coordinates passed into `get_pawn_movement` back to normal.
```py
"""In legal_nocheck:"""
...
for x, y in piece.bb.pos():
    for move in self.get_pawn_movement(x, y, piece.color):
        yield move
...
```
```py
"""In Bitboard:"""
def pos(self) -> list[Coordinate]:
    """Return the indices of any 1s in the bitboard.
    Returns:
        list[Coordinate]: The list of indices pointing to 1s.
    """

    pos = np.nonzero(self.bb)
    return list(zip(pos[1], pos[0]))
```
Output from running `board.legal_nocheck('white')`:
```
a2 -> a3
a2 -> a4
b2 -> b3
b2 -> b4
c2 -> c3
c2 -> c4
d2 -> d3
d2 -> d4
e2 -> e3
e2 -> e4
f2 -> f3
f2 -> f4
g2 -> g3
g2 -> g4
h2 -> h3
h2 -> h4
b1 -> c3
b1 -> a3
g1 -> h3
g1 -> f3
```

The legal move finder seems to be working just fine so far. The only thing left to do was eliminate moves that put the player's own king at risk of being captured.