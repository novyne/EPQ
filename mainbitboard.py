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


class Bitboard:

    def __init__(self, bb: np.ndarray | None = None) -> None:
        """Bitboard class.
        Args:
            bb (np.ndarray | None, optional): The bitboard. Defaults to None.
            x (int, optional): The width of the bitboard. Defaults to 8.
            y (int, optional): The height of the bitboard. Defaults to 8.
            \nbb argument takes priority over x and y."""
        
        self.bb = np.zeros((8, 8), dtype=int) if bb is None else bb

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        self.bb[key] = value
    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.bb[key]
    def __add__(self, other: 'Bitboard') -> 'Bitboard':
        return Bitboard(self.bb + other.bb)
    def __radd__(self, other: 'Bitboard') -> 'Bitboard':
        return Bitboard(self.bb + other.bb)
    def __invert__(self) -> 'Bitboard':
        return Bitboard(~self.bb)
    def __str__(self) -> str:
        return str(self.bb)

    def on(self) -> bool:
        """Return whether the board contains any bits."""
        return np.any(self.bb)


###################################################################################################


global PIECE_ID_INDEX
PIECE_ID_INDEX = 0

class Pawn:

    def __init__(self, color: Literal['white', 'black']) -> None:

        """Special class for pawns.
        Args:
            color (Literal['white', 'black']): The colour of the pawn.
        """

        global PIECE_ID_INDEX

        self.movement = []
        self.id = PIECE_ID_INDEX + 1
        PIECE_ID_INDEX += 1
        self.color = color

        self.bb = Bitboard()


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


class Board:

    def __init__(self) -> None:

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

        self.board = np.zeros((8, 8), dtype=int)

    def write_bitboards_from_board(self) -> None:
        """Write the piece bitboards from a board state."""
    
    def load_default_board(self) -> None:
        """Load the standard initial board into self.board."""

        struct = [
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            []
            ]


###################################################################################################


class App(tk.Tk):

    """Main application class."""

    def __init__(self) -> None:
        """Initialize the application."""

        super().__init__()

        self.title('App Template')

        self.cv = tk.Canvas(self, width=640, height=640, bg="white")
        self.cv.pack()
        
        self.s = Sprites("spritesheet.png")
        self.sprites = self.s.sprites


###################################################################################################


class Sprites:

    """Sprites class."""

    def __init__(self, path: str) -> None:
        """Initialize the sprites."""

        self.SCALE = 2

        self.spritesheet = Image.open(os.path.dirname(__file__) + '/' + path)
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


# functions go here :)


###################################################################################################


def main() -> None:
    """The main program."""

    board = Board()

    board.white.print_ids()
    board.black.print_ids()
    
    # app = App()
    # app.mainloop()

if __name__ == '__main__':
    main()

# sigma skibidi
