# default modules
import numpy as np
import os

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

    def __init__(self, bb: np.ndarray | None = None, x: int=8, y: int=8) -> None:
        """Bitboard class.
        Args:
            bb (np.ndarray | None, optional): The bitboard. Defaults to None.
            x (int, optional): The width of the bitboard. Defaults to 8.
            y (int, optional): The height of the bitboard. Defaults to 8.
            \nbb argument takes priority over x and y."""
        
        self.bb = np.zeros((x, y), dtype=int) if bb is None else bb

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


class Piece:

    def __init__(self, tag: int, color: int, movement: list[tuple[int,int]], movelong: bool) -> None:
        """Chess piece class.
        Args:
            tag (int): An identifying integer to find the piece on the board.
            color (int): The color of the piece.
        """

        self.tag = tag
        self.color = color
        self.movelong = movelong
        self.movement = self.determine_movement(movement)

        self.bb = Bitboard()
        
    def determine_movement(self, short_movement: list[tuple[int,int]]) -> list[tuple[int,int]]:
        """Extend the short list of movements into a full list that can be easily compared.
        Args:
            short_movement (list[tuple[int,int]]): The short movement list to be extended."""
    
        # add each direction
        directioned = []
        for dx, dy in short_movement:
            for _ in range(2):
                directioned.append((dx,dy))
                directioned.append((-dx,dy))
                directioned.append((dx,-dy))
                directioned.append((-dx,-dy))
                dy, dx = dx, dy
        
        if not self.movelong:
            return list(set(directioned))

        # extend in each direction
        extended = []
        for scalar in range(1,9):
            for dx, dy in list(set(directioned)):
                extended.append((dx * scalar, dy * scalar))
        
        return extended
    

class Pawn:

    def __init__(self, tag: int, color: str) -> None:
        """Special piece class for pawns.
        Args:
            tag (int): An identifying integer to find the piece on the board.
            color (str): Whether the pawn is white or black.
        """

        self.color = color
        self.movement = []



class Board:

    def __init__(self) -> None:

        class Pieces:
            def __init__(self) -> None:
                self.PAWN = Pawn(1, 'white')
                self.pawn = Pawn(2, 'black')
                self.KNIGHT = Piece(3, 'white', [(2,1)], False)
                self.knight = Piece(4, 'black', [(2,1)], False)
                self.BISHOP = Piece(5, 'white', [(1,1)], True)
                self.bishop = Piece(6, 'black', [(1,1)], True)
                self.ROOK = Piece(7, 'white', [(1,0)], True)
                self.rook = Piece(8, 'black', [(1,0)], True)
                self.QUEEN = Piece(9, 'white', [(1,0), (1,1)], True)
                self.queen = Piece(10, 'black', [(1,0), (1,1)], True)
                self.KING = Piece(11, 'white', [(1,0), (1,1)], False)
                self.king = Piece(12, 'black', [(1,0), (1,1)], False)

        self.p = Pieces()
        self.board = np.zeros((8, 8), dtype=int)
    
    def save_bitboards(self) -> None:
        """Update the bitboards for the pieces based on the board state."""

        for x, y in self.board:
            pass


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
