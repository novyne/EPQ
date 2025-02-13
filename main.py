# default modules
import cProfile
import numpy as np
import os
import random as rn

from typing import Literal, NewType, Iterable

from interface import App

try:
    # installed modules
    pass

except ModuleNotFoundError:
    print("Installing required modules from requirements.txt (sibling file). Please wait...")
    requirements = os.path.dirname(__file__) + "/requirements.txt"
    os.system("pip install -r \"" + requirements + "\"")
    print("Modules installed. Please restart the script.")
    exit()

####################################################################################################


Coordinate = NewType('Coordinate', tuple[int,int])

def cformat(x: int, y: int) -> str:
    """Format coordinates into Chess coordinates."""
    return f'{chr(x + 97)}{y + 1}'


###################################################################################################


class Bitboard:

    def __init__(self, bb: np.ndarray | None = None) -> None:
        """Bitboard class.
        Args:
            bb (np.ndarray | None, optional): The bitboard. Defaults to None.
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
        summed = Bitboard(np.sum([bb.bb for bb in other], axis=0))
        return summed

    def any(self) -> bool:
        """Return whether the board contains any bits."""
        return np.any(self.bb)
    
    def pos(self) -> list[Coordinate]:
        return [(y, x) for x, y in np.argwhere(self.bb)]


###################################################################################################


global PIECE_ID_INDEX
PIECE_ID_INDEX = -1

class Piece:

    MOVEMENT_DIRECTIONS = {
        'knight': [(2, 1), (1, 2), (-2, 1), (-1, 2), (2, -1), (1, -2), (-2, -1), (-1, -2)],
        'bishop': [(1, 1), (-1, 1), (1, -1), (-1, -1)],
        'rook': [(1, 0), (-1, 0), (0, 1), (0, -1)],
        'queen': [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)],
        'king': [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    }

    def __init__(self, color: Literal['white', 'black'], name: str, movelong: bool) -> None:
        """Chess piece class.
        Args:
            color (Literal['white', 'black']): The color of the piece.
            name (str): The name of the piece.
            movelong (bool): Whether the piece can move 'long' or not.
        """

        global PIECE_ID_INDEX
        self.id = PIECE_ID_INDEX + 1
        PIECE_ID_INDEX += 1

        self.color = color
        self.movelong = movelong
        self.movement = self.MOVEMENT_DIRECTIONS.get(name)
        self.bb = Bitboard()

    def __str__(self) -> str:
        return f"{self.color} {self.name} ID {self.id}"

    def copy(self, color: Literal['white', 'black'] = None, name: str | None = None, movelong: bool = None) -> 'Piece':
        """Create a copy of the piece, with optional overrides."""

        return Piece(
            self.color if color is None else color,
            self.name if name is None else name,
            self.movelong if movelong is None else movelong
        )
    
class Pawn(Piece):

    def __init__(self, color: Literal['white', 'black']) -> None:
        super().__init__(color, 'pawn', False)
    
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
        self.knight = Piece(color=color, name='knight', movelong=False)
        self.bishop = Piece(color=color, name='bishop', movelong=True)
        self.rook = Piece(color=color, name='rook', movelong=True)
        self.queen = Piece(color=color, name='queen', movelong=True)
        self.king = Piece(color=color, name='king', movelong=False)

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

        self.empty_piece = Piece(None, 'empty', False)
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
        
        # join IDS to pieces
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
        icon_list = ' PNBRQKpnbrqk'
        icons: dict[int, str] = {i : piece for i, piece in enumerate(icon_list)}

        white_code = '\033[91m'
        black_code = '\033[94m'
        reset_code = '\033[0m'

        s = ''
        rank_divider = '  ' + '+---' * 8 + '+\n'

        for y in range(7, -1, -1):
            rank = f"{y + 1} "

            for x in range(8):
                cell = self.board[y, x]
                icon = icons[int(cell)]
                color = white_code if icon.isupper() else black_code
                rank += f'| {color}{icon}{reset_code} '

            s += rank_divider + rank + '|\n'
        
        s += rank_divider
        s += '   ' + ' '.join(f' {chr(i + 97)} ' for i in range(8))
        return s + '\n\n'

    def copy(self) -> 'Board':
        """Return a duplicate Board instance."""
        return Board(self.turn, self.board)

    def swap_turn(self) -> 'Board':
        """Swap the turn of the board."""
        swapped_turn = self.other_player[self.turn.color]
        return Board(swapped_turn, self.board)
    
    def write_bitboards_from_board(self) -> None:
        """Write the piece bitboards from a board state."""
        id_map = {piece.id: piece for piece in self.pieces}

        non_zero_indices = np.argwhere(self.board != 0)

        for x, y in non_zero_indices:
            cell = int(self.board[x, y])
            id_map[cell].bb[x, y] = 1
    
    def update_piece_bitboard_data(self) -> None:
        """Procedure to update important information regarding the pieces and bitboards."""

        all_bb_array: np.ndarray = np.add.reduce([piece.bb.bb for piece in self.pieces], axis=0)
        self_bb_array: np.ndarray = np.add.reduce([piece.bb.bb for piece in self.self_pieces], axis=0)
        other_bb_array: np.ndarray = np.add.reduce([piece.bb.bb for piece in self.other_pieces], axis=0)

        self.all_bb = Bitboard(all_bb_array)
        self.self_bb = Bitboard(self_bb_array)
        self.other_bb = Bitboard(other_bb_array)

        self.all_pos = [(y, x) for x, y in np.argwhere(all_bb_array)]
        self.self_pos = [(y, x) for x, y in np.argwhere(self_bb_array)]
        self.other_pos = [(y, x) for x, y in np.argwhere(other_bb_array)]

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
    
    def move(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Move a piece at (x1, y1) to (x2, y2).
        Args:
            x1: The starting X coordinate.
            y1: The starting Y coordinate.
            x2: The destination X coordinate.
            y2: The destination Y coordinate.
        Returns:
            Board: The new board state.
        """

        # swap coordinates due to strange access error
        x1, y1 = y1, x1
        x2, y2 = y2, x2

        start_id = self.board[x1, y1]
        start_piece = self.id[start_id]
        end_id = self.board[x2, y2]
        end_piece = self.id[end_id]

        # on start bb: set to 0 at start and set to 1 at end
        start_piece.bb[x1, y1] = 0
        start_piece.bb[x2, y2] = 1
        # on end bb: set to 0 at end
        end_piece.bb[x2, y2] = 0

        # update board
        self.board[x1, y1] = 0
        self.board[x2, y2] = start_id

        # update last move data
        self.last_piece_taken = end_piece
        self.last_piece_moved = start_piece
        self.last_move = (x1, y1), (x2, y2)

        # update crucial bitboard data
        # self.self_bb[x1, y1] = 0
        # self.self_bb[x2, y2] = 1
        # self.self_pos = self.self_bb.pos()
        # self.other_bb[x2, y2] = 0
        # self.other_pos = self.other_bb.pos()
        # self.all_bb[x1, y1] = 0
        # self.all_bb[x2, y2] = 1
        # self.all_pos = self.all_bb.pos()

    def undo(self) -> None:
        """Undo the previous move."""

        (x1, y1), (x2, y2) = self.last_move
    
        self.last_piece_taken.bb[x2, y2] = 1
        self.board[x2, y2] = self.last_piece_taken.id

        self.last_piece_moved.bb[x2, y2] = 0
        self.last_piece_moved.bb[x1, y1] = 1
        self.board[x1, y1] = self.last_piece_moved.id

        # update crucial bitboard data
        # self.self_bb[x1, y1] = 1
        # self.self_bb[x2, y2] = 0
        # self.self_pos = self.self_bb.pos()
        # self.other_bb[x2, y2] = 1 if self.last_piece_taken.id else 0
        # self.other_pos = self.other_bb.pos()
        # self.all_bb[x1, y1] = 1
        # self.all_bb[x2, y2] = 1 if self.last_piece_taken.id else 0
        # self.all_pos = self.all_bb.pos()

    def present_pieces(self) -> list[Piece | Pawn]:
        """Get the present piece types; i.e. pieces that do not have an empty bitboard."""
        return [piece for piece in self.pieces if piece.bb.any() and piece.id != 0]

    def get_pawn_movement(self, x: int, y: int) -> list[list[Coordinate, Coordinate]]:
        """Obtain the possible movement of a pawn based on its position and colour.
        Args:
            x (int): The X coordinate of the pawn.
            y (int): The Y coordinate of the pawn.
        Returns:
            list(list[Coordinate, Coordinate]): The list of coordinates denoting where the pawn is allowed to move.
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
            Iterable(list[Coordinate, Coordinate]): An iterable of pairs of coordinates describing the movement.
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
            Iterable(list[Coordinate, Coordinate]): An iterable of pairs of coordinates describing the movement.
        """

        moveable_pieces = list(set(self.present_pieces()) & set(self.self_pieces))

        for piece in moveable_pieces:
            if isinstance(piece, Pawn):
                for x, y in piece.bb.pos():
                    for move in self.get_pawn_movement(x, y):
                        yield move
                pass
            elif isinstance(piece, Piece):
                for move in list(self.piece_legal_nocheck(piece)):
                    yield move

    def legal_moves(self) -> Iterable[list[Coordinate, Coordinate]]:
        """Find all legal moves on the board.
        Returns:
            Iterable(list[Coordinate, Coordinate]): The legal moves in pairs of coordinates.
        """

        legals_nocheck = self.legal_nocheck()

        for [(x1, y1), (x2, y2)] in legals_nocheck:
            self.move(x1, y1, x2, y2)
            if not self.isking_vulnerable():
                yield [(x1, y1), (x2, y2)]
            self.undo()

    def isking_vulnerable(self) -> bool:
        """Return whether the turn player's king can be taken.
        Returns:
            bool: True if the king can be taken, otherwise False.
        """

        # obtain the legal moves for the other player
        other_board = self.swap_turn()
        other_legals_nocheck = other_board.legal_nocheck()

        for [(x1, y1), (x2, y2)] in other_legals_nocheck:
            other_board.move(x1, y1, x2, y2)
            king = other_board.other_player[other_board.turn.color].king # obtain the CURRENT player's king

            # if the king is not present, return True
            if not king.bb.any():
                other_board.undo()
                return True
            
            other_board.undo()
        return False


###################################################################################################


def random_game() -> None:
    """Continually play random moves until one computer runs out of legal moves."""

    custom_board = [[10, 8, 9, 11, 12, 9, 8, 10],
                    [0] * 8,
                    [0] * 8,
                    [0] * 8,
                    [0] * 8,
                    [0] * 8,
                    [0] * 8,
                    [4, 2, 3, 5, 6, 3, 2, 4]
                    ][::-1]

    # board = Board().default()
    board = Board(board=np.array(custom_board))

    while True:
        legals = list(board.legal_moves())

        if not legals:
            print(f"No legal moves left for {board.turn.color}. Game over!")
            break

        random_move = rn.choice(legals)
        [(x1, y1), (x2, y2)] = random_move
        print(f"{board.turn.color} moves: {chr(x1 + 97)}{y1+1} -> {chr(x2 + 97)}{y2+1}")

        board.move(x1, y1, x2, y2)
        board = board.swap_turn()

        print(board)


###################################################################################################


def main() -> None:
    """The main program."""

    random_game()
    
    # app = App()
    # app.mainloop()

if __name__ == '__main__':
    cProfile.run('main()',sort='cumulative')