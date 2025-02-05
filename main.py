# default modules
import numpy as np
import os
import random as rn

from typing import Literal, NewType, Iterable

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


Coordinate = NewType('Coordinate', tuple[int,int])


###################################################################################################


class Bitboard:

    def __init__(self, bb: np.ndarray | None = None) -> None:
        """Bitboard class.
        Args:
            bb (np.ndarray | None, optional): The bitboard. Defaults to None.
            x (int, optional): The width of the bitboard. Defaults to 8.
            y (int, optional): The height of the bitboard. Defaults to 8.
            \nbb argument takes priority over x and y.
        """
        
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
    
    def sum(self, other) -> 'Bitboard':
        summed = Bitboard(self.bb)
        for bb in other:
            summed += bb
        return summed

    def any(self) -> bool:
        """Return whether the board contains any bits."""
        return np.any(self.bb)
    
    def pos(self) -> list[Coordinate]:
        """Return the indices of any 1s in the bitboard.
        Returns:
            list[Coordinate]: The list of indices pointing to 1s.
        """

        pos = np.nonzero(self.bb)
        return list(zip(pos[1], pos[0]))


###################################################################################################


global PIECE_ID_INDEX
PIECE_ID_INDEX = -1

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

    def __str__(self) -> str:
        return f"{self.color} piece ID {self.id}"

    def expand_movement(self, short_movement: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Expand the short movement list for all cases of movement.
        Args:
            short_movement (list[Coordinate]): The short-movement list.
        Returns:
            list[Coordinate]: The expanded movement list.
        """

        mvmt = []
        for dx, dy in short_movement:
            for _ in range(2):
                mvmt.append((dx, dy))
                mvmt.append((dx, -dy))
                mvmt.append((-dx, dy))
                mvmt.append((-dx, -dy))
                dy, dx = dx, dy

        return mvmt

    def copy(self, color: Literal['white', 'black'] = None, movement: list[tuple[int, int]] = None, movelong: bool = None) -> 'Piece':
        """Create a copy of the piece, with optional overrides."""

        return Piece(
            self.color if color is None else color,
            self.movement if movement is None else movement,
            self.movelong if movelong is None else movelong
        )
    
class Pawn(Piece):

    def __init__(self, color: Literal['white', 'black']) -> None:
        super().__init__(color, [], False)
    
    def copy(self, color: Literal['white', 'black'] = None) -> 'Pawn':
        """Create a copy of the pawn, with optional overrides."""

        return Pawn(
            self.color if color is None else color
        )


class Player:

    def __init__(self, color: Literal['white', 'black']) -> None:
        """Player class to manage pieces for a given colour.
        Args:
            color (Literal['white', 'black']): The colour of the piece.
        """

        self.color: Literal['white','black'] = color

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

class Board:

    def __init__(self, turn: Player | None = None, board: np.ndarray | None = None) -> None:
        """The class for the board.
        Args:
            turn (Player, optional): The player of whom it is their turn to move.
            board (np.array, optional): A pre-existing array the board can take on. Defaults to an empty board if unspecified.
        """

        global PIECE_ID_INDEX
        PIECE_ID_INDEX = -1

        self.empty_piece = Piece(None, [(0,0)], False)
        self.white = Player('white')
        self.black = Player('black')
        self.other_player = {
            'white' : self.black,
            'black' : self.white
        }

        self.turn = self.white if turn is None else turn

        self.pieces: list[Piece] = [self.empty_piece] + self.white.pieces.copy() + self.black.pieces.copy()

        self.board = np.zeros((8, 8), dtype=int) if board is None else board
        # board is an array of piece IDs
        self.write_bitboards_from_board()
        
        self.id: dict[int, Piece | Pawn] = {}
        for piece in self.pieces:
            self.id[piece.id] = piece
    
        self.last_piece_taken: None | Piece | Pawn = None
        self.last_piece_moved: None | Piece | Pawn = None
        self.last_move: None | tuple[Coordinate, Coordinate] = None
        
        self.self_pieces = self.white.pieces if self.turn.color == 'white' else self.black.pieces
        self.other_pieces = self.black.pieces if self.turn.color == 'white' else self.white.pieces

        self.update_piece_bitboard_data()

    def __str__(self) -> str:
        s = ''
        for x in range(7, -1, -1):
            for y in self.board[x]:
                s += str(y) + '\t'
            s += '\n'
        return s

    def copy(self) -> 'Board':
        """Return a duplicate Board instance."""
        return Board(self.turn, self.board)

    def swap_turn(self) -> 'Board':
        """Swap the turn of the board."""
        swapped_turn = self.other_player[self.turn.color]
        return Board(swapped_turn, self.board) 

    def write_bitboards_from_board(self) -> None:
        """Write the piece bitboards from a board state."""

        id = {piece.id : piece for piece in self.pieces}

        for x in range(8):
            for y in range(8):
                cell = int(self.board[x, y])
                if cell == 0: continue
                id[cell].bb[x, y] = 1
    
    def update_piece_bitboard_data(self) -> None:
        """Procedure to update important information regarding the pieces and bitboards."""

        self.self_bb = Bitboard().sum([piece.bb for piece in self.self_pieces])
        self.self_pos = self.self_bb.pos()
        self.other_bb = Bitboard().sum([piece.bb for piece in self.other_pieces])
        self.other_pos = self.other_bb.pos()
        self.all_bb = Bitboard().sum([piece.bb for piece in self.pieces])
        self.all_pos = self.all_bb.pos()

    def default(self) -> 'Board':
        """Return the default Chess board in piece IDs.
        Returns:
            Board: The board instance.
        """

        black = self.black
        white = self.white

        struct: list[list[Piece | Pawn | Literal[0]]] = [
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
                if piece == 0:
                    row_ids.append(0)
                else:
                    row_ids.append(piece.id)
            struct_ids.append(row_ids)
        
        board = np.array(struct_ids)
        return Board(board=board)

    def display_bitboards(self) -> None:
        """Print each bitboard for each piece."""

        for piece in self.pieces:
            print(f"ID: {piece.id}")
            print(piece.bb)

    def move(self, x1: int, y1: int, x2: int, y2: int) -> 'Board':
        """Move a piece at (x1, y1) to (x2, y2).
        Args:
            x1, y1: The coordinates of the start.
            x2: y2: The coordinates of the end.
        Returns:
            Board: The new board state.
        """

        new = self.copy()

        # swap coordinates due to strange access error
        x1, y1 = y1, x1
        x2, y2 = y2, x2

        start_id = new.board[x1, y1]
        start_piece = new.id[start_id]
        end_id = new.board[x2, y2]
        end_piece = new.id[end_id]

        # on start bb: set to 0 at start and set to 1 at end
        start_piece.bb[x1, y1] = 0
        start_piece.bb[x2, y2] = 1
        # on end bb: set to 0 at end
        end_piece.bb[x2, y2] = 0

        # update board
        new.board[x1, y1] = 0
        new.board[x2, y2] = start_id

        # update last move data
        new.last_piece_taken = end_piece
        new.last_piece_moved = start_piece
        new.last_move = (x1, y1), (x2, y2)

        # switch turn
        new.turn = new.other_player[new.turn.color]

        # update crucial bitboard data
        new.update_piece_bitboard_data()

        return new

    def undo(self) -> None:
        """Undo the previous move.
        """

        (x1, y1), (x2, y2) = self.last_move
    
        self.last_piece_taken.bb[x2, y2] = 1
        self.board[x2, y2] = self.last_piece_taken.id

        self.last_piece_moved.bb[x2, y2] = 0
        self.last_piece_moved.bb[x1, y1] = 1
        self.board[x1, y1] = self.last_piece_moved.id

    def present_pieces(self) -> list[Piece | Pawn]:
        """Get the present piece types; i.e. pieces that do not have an empty bitboard."""
        return [piece for piece in self.pieces if piece.bb.any()]

    def get_pawn_movement(self, x: int, y: int) -> list[list[Coordinate, Coordinate]]:
        """Obtain the possible movement of a pawn based on its position and colour.
        Args:
            x (int): The X coordinate of the pawn.
            y (int): The Y coordinate of the pawn.
        """

        movement: list[list[Coordinate, Coordinate]] = []

        # y direction to check in
        if self.turn.color == 'white':
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
        if self.turn.color == 'white' and y == 1:
            movement.append([(x, y), (x, y+(dy*2))])
        elif self.turn.color == 'black' and y == 6:
            movement.append([(x, y), (x, y+(dy*2))])
        return movement

    def piece_legal_nocheck(self, piece: Piece) -> Iterable[list[Coordinate, Coordinate]]:
        """Get all legal moves for a single piece, specified by class instance.
        Args:
            piece (Piece): The piece to check for legal moves.
        Returns:
            Iterable[list[Coordinate, Coordinate]]: An iterable of pairs of coordinates describing the movement.
        """

        for x, y in piece.bb.pos():
            for dx, dy in piece.movement:
                rx, ry = x + dx, y + dy

                # cannot 'capture' own piece
                if (rx, ry) in self.self_pos:
                    continue
                # out of bounds
                if rx not in range(8) or ry not in range(8):
                    continue

                # cannot movelong
                if not piece.movelong:
                    yield [(x, y), (rx, ry)]
                    continue

                for s in range(1, 9):
                    # scaled direction and resultant X and Y
                    sdx, sdy = dx * s, dy * s
                    rx, ry = x + sdx, y + sdy

                    # cannot be out of bounds
                    if rx not in range(8) or ry not in range(8):
                        break

                    # cannot 'capture' own piece
                    if (rx, ry) in self.self_pos:
                        break
                    # if capturing enemy piece, yield then break
                    if (rx, ry) in self.other_pos:
                        yield [(x, y), (rx, ry)]
                        break

                    # otherwise yield
                    yield [(x, y), (rx, ry)]

    def legal_nocheck(self) -> Iterable[list[Coordinate, Coordinate]]:
        """Get all legal moves without checking for checks.
        Returns:
            Iterable[list[Coordinate, Coordinate]]: An iterable of pairs of coordinates describing the movement.
        """

        moveable_pieces = list(set(self.present_pieces()) & set(self.self_pieces))

        for piece in moveable_pieces:
            if isinstance(piece, Pawn):
                for x, y in piece.bb.pos():
                    for move in self.get_pawn_movement(x, y):
                        yield move
            elif isinstance(piece, Piece):
                for move in self.piece_legal_nocheck(piece):
                    yield move

    def legal_moves(self) -> Iterable[list[Coordinate, Coordinate]]:
        """Find all legal moves on the board.
        Returns:
            Iterable[list[Coordinate, Coordinate]]: The legal moves in pairs of coordinates.
        """

        legals_nocheck = self.legal_nocheck()

        for [(x1, y1), (x2, y2)] in legals_nocheck:
            board = self.move(x1, y1, x2, y2)
            if not board.isking_vulnerable():
                yield [(x1, y2), (x2, y2)]
            board.undo()

    def isking_vulnerable(self) -> bool:
        """Return whether the turn player's king can be taken.
        Returns:
            bool: True if the king can be taken, otherwise False.
        """

        # obtain the legal moves for the other player
        other_board = self.swap_turn()
        other_legals_nocheck = other_board.legal_nocheck()

        for [(x1, y1), (x2, y2)] in other_legals_nocheck:
            board = other_board.move(x1, y1, x2, y2)
            king = board.other_player[board.turn.color].king # obtain the CURRENT player's king
            
            # if the king is not present, return True
            if not king.bb.any():
                return True
            
            board.undo()
        return False

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

    POSITIONS = {

    }

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





###################################################################################################


def main() -> None:
    """The main program."""

    board = Board().default()

    legals = list(board.legal_moves())

    for m1, m2 in legals:
        print(f"{chr(m1[0] + 97)}{m1[1] + 1} -> {chr(m2[0] + 97)}{m2[1] + 1}")

    print(board)
    
    # app = App()
    # app.mainloop()

if __name__ == '__main__':
    main()