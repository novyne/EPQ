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


# globals and pre-main functions go here :)


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


global PIECE_TAG_INDEX

PIECE_TAG_INDEX = 0


class Pawn:

    def __init__(self, color: Literal['white', 'black']) -> None:

        """Special class for pawns.
        Args:
            color (Literal['white', 'black']): The colour of the pawn.
        """

        global PIECE_TAG_INDEX

        self.movement = []
        self.tag = PIECE_TAG_INDEX + 1
        PIECE_TAG_INDEX += 1
        self.color = color


class Piece:

    def __init__(self, color: Literal['white', 'black'], movement: list[tuple[int,int]], movelong: bool) -> None:
        """Chess piece class.
        Args:
            color (Literal['white', 'black']): The color of the piece.
            movement (list[tuple[int,int]]): The short-hand movement of the piece.
            movelong (bool): Whether the piece can move 'long' or not.
        """

        global PIECE_TAG_INDEX

        self.tag = PIECE_TAG_INDEX + 1
        PIECE_TAG_INDEX += 1

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

        class Pieces:

            def __init__(self) -> None:

                self.PAWN = Pawn('white')
                self.pawn = Pawn('black')
                self.KNIGHT = Piece('white', [(2,1)], False)
                self.knight = self.KNIGHT.copy(color='black')
                self.BISHOP = Piece('white', [(1,1)], True)
                self.bishop = self.BISHOP.copy(color='black')
                self.ROOK = Piece('white', [(1,0)], True)
                self.rook = self.ROOK.copy(color='black')
                self.QUEEN = Piece('white', [(1,0), (1,1)], True)
                self.queen = self.QUEEN.copy(color='black')
                self.KING = self.QUEEN.copy(movelong=False)
                self.king = self.queen.copy(movelong=False)

        self.p = Pieces()
        self.board = np.zeros((8, 8), dtype=int)

    def write_bitboards_from_board(self) -> None:
        """Write the piece bitboards from a board state."""


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

    print(board.p.QUEEN.movement)

    # app = App()
    # app.mainloop()

if __name__ == '__main__':
    main()

# sigma skibidi
