# default modules
import cProfile
import numpy as np
import os
import random as rn

from typing import Literal, NewType, Iterable, Optional

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
Move = NewType('Move', list[Coordinate, Coordinate, Optional[int]])

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
    
    def copy(self) -> 'Bitboard':
        return Bitboard(self.bb.copy())


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

        self.pieces: list[Piece] = [self.pawn, self.knight, self.bishop, self.rook, self.queen, self.king]

    def print_ids(self) -> None:
        """Print the IDs of each piece for this player."""
        names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        for name, piece in zip(names, self.pieces):
            print(f"{self.color} {name} : {piece.id}")

class Board:

    """Chess board class."""

    def __init__(self, turn: Optional[Player] = None, board: Optional[np.ndarray] = None) -> None:
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

        self.turn = turn or self.white

        self.pieces: list[Piece] = [self.empty_piece] + self.white.pieces + self.black.pieces
        self.board = np.zeros((8,8), dtype=int) if board is None else board

        # board is an array of piece IDs
        self.write_bitboards_from_board()
        
        # join IDS to pieces
        self.id: dict[int, Piece | Pawn] = {piece.id : piece for piece in self.pieces}

        self.last_pieces_taken: list[Piece | Pawn] = []
        self.last_pieces_moved: list[Piece | Pawn] = []
        self.last_moves: list[tuple[Coordinate, Coordinate]] = []

        if self.turn.color == 'white':
            self.self_pieces = self.white.pieces
            self.other_pieces = self.black.pieces
        else:
            self.self_pieces = self.black.pieces
            self.other_pieces = self.white.pieces

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

    def swap_turn_with_new_instance(self) -> 'Board':
        """Like swap_turn, but returns a new instance instead."""

        swapped_turn = self.other_player[self.turn.color]
        return Board(swapped_turn, self.board)
    
    def swap_turn(self) -> None:
        """Changes the turn on the board and swaps necessary variables."""
        swapped_turn = self.other_player[self.turn.color]
        self.turn = swapped_turn

        self.self_bb, self.other_bb = self.other_bb.copy(), self.self_bb.copy()
        self.self_pieces, self.other_pieces = self.other_pieces.copy(), self.self_pieces.copy()
        self.self_pos, self.other_pos = self.other_pos.copy(), self.self_pos.copy()

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
    
    def move(self, x1: int, y1: int, x2: int, y2: int, promotion: Optional[int] = None) -> None:
        """Move a piece at (x1, y1) to (x2, y2).
        Args:
            x1 (int): The starting X coordinate.
            y1 (int): The starting Y coordinate.
            x2 (int): The destination X coordinate.
            y2 (int): The destination Y coordinate.
            promotion (int, optional): The ID of the piece to promote to if applicable.
        Returns:
            Board: The new board state.
        """

        # swap coordinates due to strange access error
        x1, y1 = y1, x1
        x2, y2 = y2, x2

        if self.board[x1, y1] == 0:
            raise ValueError(f"""Cannot move empty square. ({x1} {y1} to {x2} {y2})\n\n{self}""")

        start_id = promotion or self.board[x1, y1]
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
        self.last_pieces_taken.append(end_piece)
        if promotion is None:
            self.last_pieces_moved.append(self.id[self.board[x2, y2]])
        else:
            self.last_pieces_moved.append(self.id[promotion])
        self.last_moves.append(((x1, y1), (x2, y2)))

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

        (x1, y1), (x2, y2) = self.last_moves.pop()
        last_piece_taken = self.last_pieces_taken.pop()
        last_piece_moved = self.last_pieces_moved.pop()
    
        last_piece_taken.bb[x2, y2] = 1
        self.board[x2, y2] = last_piece_taken.id

        last_piece_moved.bb[x2, y2] = 0
        last_piece_moved.bb[x1, y1] = 1
        self.board[x1, y1] = last_piece_moved.id

        # update crucial bitboard data
        # self.self_bb[x1, y1] = 1
        # self.self_bb[x2, y2] = 0
        # self.self_pos = self.self_bb.pos()
        # self.other_bb[x2, y2] = 1 if self.last_piece_taken.id else 0
        # self.other_pos = self.other_bb.pos()
        # self.all_bb[x1, y1] = 1
        # self.all_bb[x2, y2] = 1 if self.last_piece_taken.id else 0
        # self.all_pos = self.all_bb.pos()

    # def present_pieces(self) -> list[Piece | Pawn]:
    #     """Get the present piece types; i.e. pieces that do not have an empty bitboard."""
    #     return [piece for piece in self.pieces if piece.bb.any() and piece.id != 0]

    def get_pawn_movement(self, x: int, y: int) -> Iterable[Move]:
        """Obtain the possible movement of a pawn based on its position and colour.
        Args:
            x (int): The X coordinate of the pawn.
            y (int): The Y coordinate of the pawn.
        Returns:
            Iterable(list):
                Coordinate: The starting coordinate.

                Coordinate: The end coordinate.

                Optional[int]: The optional promotion ID.
        """

        def legal_pawn_no_promotion() -> Iterable[list[Coordinate]]:
            """Helper function."""

            # y direction to check in
            dy = 1 if self.turn.color == 'white' else -1

            # capture
            for dx in (1, -1):
                if (x+dx, y+dy) in self.other_pos:
                    yield [(x, y), (x+dx, y+dy)]

            # check if the square ahead is occupied
            if (x, y+dy) in self.all_pos:
                return
            yield [(x, y), (x, y+dy)]

            # check whether the pawn can move 2 squares forward
            if (x, y+(dy*2)) in self.all_pos:
                return
            if self.turn.color == 'white' and y == 1:
                yield [(x, y), (x, y+(dy*2))]
            elif self.turn.color == 'black' and y == 6:
                yield [(x, y), (x, y+(dy*2))]
            return
        
        prom_rank = 7 if self.turn.color == 'white' else 0
        promotions = [2,3,4,5] if self.turn.color == 'white' else [8,9,10,11]

        for [(x1,y1),(x2,y2)] in legal_pawn_no_promotion():
            if y2 == prom_rank:
                for p in promotions:
                    yield [(x1,y1),(x2,y2), p]
            else:
                yield [(x1,y1),(x2,y2), None]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < 8 and 0 <= y < 8

    def piece_legal_nocheck(self, piece: Piece) -> Iterable[Move]:
        """Get all legal moves for a single piece, specified by class instance.
        Args:
            piece (Piece): The piece to check for legal moves.
        Returns:
            Iterable(Move): An iterable of possible moves.
        """

        for x, y in piece.bb.pos():
            for dx, dy in piece.movement:
                rx, ry = x + dx, y + dy

                # cannot 'capture' own piece or move out of bounds
                if not self.in_bounds(rx, ry) or (rx, ry) in self.self_pos:
                    continue

                # cannot movelong
                if not piece.movelong:
                    yield [(x, y), (rx, ry), None]
                    continue

                for s in range(1, 9):
                    # scaled direction and resultant X and Y
                    sdx, sdy = dx * s, dy * s
                    rx, ry = x + sdx, y + sdy

                    # cannot 'capture' own piece or move out of bounds
                    if not self.in_bounds(rx, ry) or (rx, ry) in self.self_pos:
                        break
                    # if capturing enemy piece, yield then break
                    if (rx, ry) in self.other_pos:
                        yield [(x, y), (rx, ry), None]
                        break

                    # otherwise yield
                    yield [(x, y), (rx, ry), None]
    
    def new_piece_legal_nocheck(self, piece: Piece) -> Iterable[Move]:
        """Get all legal moves for a single piece, specified by class instance.
        Args:
            piece (Piece): The piece to check for legal moves.
        Returns:
            Iterable(Move): An iterable of moves.
        """

        for x, y in piece.bb.pos():
            for dx, dy in piece.movement:
                for s in range(9):
                    # scaled direction and resultant X and Y
                    sdx, sdy = dx * s, dy * s
                    rx, ry = x + sdx, y + sdy

                    # cannot 'capture' own piece or move out of bounds
                    if not self.in_bounds(rx, ry) or (rx, ry) in self.self_pos:
                        break

                    # cannot movelong
                    if not piece.movelong:
                        yield [(x, y), (rx, ry), None]
                        break

                    # if capturing enemy piece, yield then break
                    if (rx, ry) in self.other_pos:
                        yield [(x, y), (rx, ry), None]
                        break

                    # otherwise yield
                    yield [(x, y), (rx, ry), None]

    def legal_nocheck(self) -> Iterable[Move]:
        """Get all legal moves without checking for checks.
        Returns:
            Iterable(Move): An iterable of moves.
        """

        for piece in (p for p in self.self_pieces if p.bb.any()):
            if isinstance(piece, Pawn):
                for x, y in piece.bb.pos():
                    yield from self.get_pawn_movement(x, y)
            else:
                yield from self.piece_legal_nocheck(piece)

    def legal_moves(self) -> Iterable[Move]:
        """Find all legal moves on the board.
        Returns:
            Iterable(Move): The legal moves.
        """

        legals_nocheck = self.legal_nocheck()

        for [(x1, y1), (x2, y2), promotion] in list(legals_nocheck):
            self.move(x1, y1, x2, y2)
            if not self.isking_vulnerable():
                yield [(x1, y1), (x2, y2), promotion]
            self.undo()
    
    def isking_vulnerable(self) -> bool:
        """Return whether the turn player's king can be taken.
        Returns:
            bool: True if the king can be taken, otherwise False.
        """

        # obtain the legal moves for the other player
        self = self.swap_turn_with_new_instance()
        # self.swap_turn()
        other_legals_nocheck = self.legal_nocheck()

        for [(x1, y1), (x2, y2), promotion] in other_legals_nocheck:
            self.move(x1, y1, x2, y2, promotion)
            king = self.other_player[self.turn.color].king # obtain the CURRENT player's king

            # if the king is not present, return True
            if not king.bb.any():
                self.undo()
                # self.swap_turn()
                self = self.swap_turn_with_new_instance()
                return True
            
            self.undo()
        # self.swap_turn()
        self = self.swap_turn_with_new_instance()
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

    board = Board().default()
    # board = Board(board=np.array(custom_board))

    while True:
        legals = list(board.legal_moves())

        if not legals:
            print(f"No legal moves left for {board.turn.color}. Game over!")
            break

        random_move = rn.choice(legals)
        [(x1, y1), (x2, y2), promotion] = random_move
        print(f"{board.turn.color} moves: {chr(x1 + 97)}{y1+1} -> {chr(x2 + 97)}{y2+1}")

        board.move(x1, y1, x2, y2, promotion)
        board = board.swap_turn_with_new_instance()

        print(board)
        # input()


###################################################################################################


def main() -> None:
    """The main program."""

    random_game()
    # board = Board().default()

    # app = App()
    # app.mainloop()

if __name__ == '__main__':
    main()