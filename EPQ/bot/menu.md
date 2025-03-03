# EPQ Menu - The Computer

## Framework

### Overview

To create this computer, I decided to use a typical approach for board game computers - **Minmaxing**. The process is as follows:

* Use a recursive method to 'search' every move `n` moves in the future
* Score the board position with an algorithm
* Apply a *minmaxing* algorithm during the search to decide the move required to maximise / minimise the best possible score.

I first started with a file template:

```py
# default modules
import os

from main import Board

try:
    # installed modules
    open(os.path.dirname(__file__) + '/requirements.txt', 'w').close()

except ModuleNotFoundError:
    print("Installing required modules from requirements.txt (sibling file). Please wait...")
    requirements = os.path.dirname(__file__) + "/requirements.txt"
    os.system("pip install -r \"" + requirements + "\"")
    print("Modules installed. Please restart the script.")
    exit()


####################################################################################################


# Globals & Post-class Functions


###################################################################################################


# Classes


###################################################################################################


# Post-class Functions


###################################################################################################


def main() -> None:
    """The main program."""

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nExecution interrupted.")
```

Firstly, I created a computer class, taking just the board as a parameter.

```py
class Computer:

    def __init__(self, board: Board) -> None:
        self.board = board


###################################################################################################


# Post-class Functions


###################################################################################################


def main() -> None:
    """The main program."""

    board: Board = Board().default()
    computer = Computer(board)
```

Here is a reminder of the plan to create the backbone of the computer.

* Use a recursive method to 'search' every move `n` moves in the future
* Score the board position with an algorithm
* Apply a *minmaxing* algorithm during the search to decide the move required to maximise / minimise the best possible score.

I then designed a function for each, providing most of the parameters, returns, and docnotes.

```py
class Computer:

    def __init__(self, board: Board) -> None:
        self.board = board

    def minmax(self, remaining_depth: int) -> float:
        """Use minmax to score a position by searching through a movetree down a given depth.
        Args:
            remaining_depth (int): The remaining depth of the search.
        Returns:
            float: The score.
        """
    
    def evaluate(self) -> float:
        """Score the current board position.
        Returns:
            float: The score.
        """
```

The first part of the plan would be to create the recursive search.

### The Search Algorithm

I designed a plan to build the function:

1. Collect and iterate through legal moves, trying each one
2. Re-call the function with a lower depth
3. Stop the recursion if depth reaches 0 and instead return the evaluated score
4. Collect the scores into a list
5. Undo the move once the function completes
6. Apply minmax to the scores.

```py
def minmax(self, remaining_depth: int, board: Board) -> float:
    """Use minmax to score a position by searching through a movetree down a given depth.
    Args:
        remaining_depth (int): The remaining depth of the search.
        board (Board): The board state to apply minmax to.
    Returns:
        float: The score.
    """

    # Step 3
    if remaining_depth == 0:
        return self.evaluate(board) # Added a board parameter for ease
    
    # Step 4.1
    scores = []
    
    # Step 1
    for [(x1, y1), (x2, y2)] in board.legal_moves():
        board.move(x1, y1, x2, y2)

        # Step 2
        score = self.minmax(remaining_depth - 1, board.swap_turn())
        # Step 4.2
        scores.append(score)

        # Step 5
        board.undo()
    
    # Step 6
    ismaximising = self.board.turn.color != 'white'
    return max(scores) if ismaximising else min(scores)
```

Notice how I added the `Board` parameter. This makes swapping the turn and making moves a lot easier, avoiding unecessary copying.

#### Tests

To test the function, I designed a primitive evaluation function, which simply compares material advantage. I also added noise to ensure the output wouldn't always be the same.

```py
class Computer:

    def __init__(self, board: Board) -> None:
        """Computer class. Evaluates the 'best' move from a given Board instance.
        Args:
            board (Board): The board instance to evaluate.
        """
        
        self.board = board

        self.material = [0] + [1, 3, 3.25, 5, 9, float('inf')] * 2

    def minmax(self, remaining_depth: int, board: Board) -> float:
        ...
    
    def evaluate(self, board: Board) -> float:
        """Score a given board position.
        Args:
            board (Board): The board instance to score.
        Returns:
            float: The score.
        """

        score = rn.random() / 10

        for piece in board.white.pieces[:-1]:
            count = piece.bb.bb.sum()
            count *= self.material[piece.id]

            score += count

        for piece in board.black.pieces[:-1]:
            count = piece.bb.bb.sum()
            count *= self.material[piece.id]

            score -= count

        return score
```

I printed the evaluation of the board as well as the board following the move. Here is an extract of the last few moves:

```plaintext
-19.582024919860466
  +---+---+---+---+---+---+---+---+
8 |   | n | b | q |   | b |   | r |
  +---+---+---+---+---+---+---+---+
7 |   |   |   |   |   |   |   | p |
  +---+---+---+---+---+---+---+---+
6 |   | p |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
4 | P |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
2 |   |   |   |   |   | R | K |   |
  +---+---+---+---+---+---+---+---+
1 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h 


-19.447929356538204
  +---+---+---+---+---+---+---+---+
8 |   | n | b | q |   | b |   | r |
  +---+---+---+---+---+---+---+---+
7 |   |   |   |   | n |   |   | p |
  +---+---+---+---+---+---+---+---+
6 |   | p |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
2 |   |   |   |   |   | R | K |   |
  +---+---+---+---+---+---+---+---+
1 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h 
```

Pieces are vanishing and reappearing out of nowhere. Clearly, the moves aren't being undone correctly.

To better assess the problem, I printed the board before and after the undo.

It seemed as though when a piece or pawn moved, it simply vanished. I decided to run `board.move` in its native script. Strangely enough, it worked without any issues.

I then decided to print the board before the legal move finder, then just before the move. Here was the result:

```plaintext
  +---+---+---+---+---+---+---+---+
8 | r | n | b | q | k | b | n | r |
  +---+---+---+---+---+---+---+---+
7 | p | p | p | p | p | p | p | p |
  +---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
2 | P | P | P | P | P | P | P | P |
  +---+---+---+---+---+---+---+---+
1 | R | N | B | Q | K | B | N | R |
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h 


  +---+---+---+---+---+---+---+---+
8 | r | n | b | q | k | b | n | r |
  +---+---+---+---+---+---+---+---+
7 | p | p | p | p | p | p | p | p |
  +---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
3 | P |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
2 |   | P | P | P | P | P | P | P |
  +---+---+---+---+---+---+---+---+
1 | R | N | B | Q | K | B | N | R |
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h 

```

The reason that pieces were disappearing was because the move algorithm was trying to move an empty piece. To prevent this in the future, I added a ValueError guard at the top of `Board.move` to stop the execution if it occurs.

```py
"""In board.main:board.move:"""

...
x2, y2 = y2, x2

start_id = self.board[x1, y1]

if start_id == 0:
    raise ValueError("""Cannot move empty square. (ID: 0)""")

start_piece = self.id[start_id]
end_id = self.board[x2, y2]
end_piece = self.id[end_id]
...
```

I then assumed there was a problem with the generator. Converting the legal moves to a list as below solved the issue.

```py
"""In Computer.minmax:"""
...
for [(x1, y1), (x2, y2)] in list(board.legal_moves()):
    ...
```

Now, the function was working perfectly fine.

### The Best Move

All that was left was to convert the evaluation into what the computer assumed to be the best move.

```py
def find_best_move(self, depth: int) -> tuple[int,int,int,int]:
    """Determine the best move based on the evaluation and minmaxing.
    Args:
        depth (int): The depth of the movetree to search down.
    Returns:
        tuple(int, int, int, int): The four indices (x1, y1, x2, y2) pointing to the best move.
    """

    ismaximising = self.board.turn.color == 'white'
    best_move = None
    best_score = -float('inf') if ismaximising else float('inf')

    for [(x1, y1), (x2, y2)] in list(self.board.legal_moves()):
        self.board.move(x1, y1, x2, y2)
        score = self.minmax(depth - 1, self.board.swap_turn())

        if (ismaximising and score > best_score) or (not ismaximising and score < best_score):
            best_score = score
            best_move = (x1, y1, x2, y2)
        
        self.board.undo()
    
    return best_move
```

The output:

```plaintext
4 1 to 4 3
```

Unfortunately, a simple low-depth search with a simple evaluation function took several seconds, so I would have to do a lot of optimisation.

## Optimisation

I first wrote a small function to test the computer:

```py
def main() -> None:
    """The main program."""

    board: Board = Board().default()

    for _ in range(5):
        computer = Computer(board)
        x1, y1, x2, y2 = computer.find_best_move(depth=2)
        board.move(x1, y1, x2, y2)
        board = board.swap_turn()
        print(board)
```

I then profiled it using `time_test.py`.

Here are the top 10 culprits:

```bash
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        5   0.0012   0.0002   3.5655   0.7131 main.py:90(find_best_move)
 3352/124   0.0238   0.0000   3.4390   0.0277 main.py:41(minmax)
     3481   0.0222   0.0000   2.6466   0.0008 main.py:465(legal_moves)
     3358   0.1147   0.0000   2.5746   0.0008 main.py:479(isking_vulnerable)
     6715   0.0067   0.0000   1.5252   0.0002 main.py:230(swap_turn)
     6717   0.0341   0.0000   1.5190   0.0002 main.py:158(__init__)
    98546   0.0636   0.0000   0.8540   0.0000 main.py:452(legal_nocheck)
     6717   0.4411   0.0001   0.7513   0.0001 main.py:245(update_piece_bitboard_data)
   112607   0.0930   0.0000   0.5531   0.0000 main.py:63(any)
     6717   0.3902   0.0001   0.5174   0.0001 main.py:235(write_bitboards_from_board)  
```

### `Board.swap_turn`

I decided to create a new `Board.swap_turn` that instead of returning a new `Board` instance, edited the current instance.

```py
"""In board.main:Board:"""

def swap_turn_on_current(self) -> None:
    """Like Board.swap_turn, but edits the current instance instead of returning a new one."""
    swapped_turn = self.other_player[self.turn.color]
    self.turn = swapped_turn
```

I then modified `Computer.minmax` to account for this change:

```py
def minmax(self, remaining_depth: int, board: Board) -> float:
    """Use minmax to score a position by searching through a movetree down a given depth.
    Args:
        remaining_depth (int): The remaining depth of the search.
        board (Board): The board state to apply minmax to.
    Returns:
        float: The score.
    """

    if remaining_depth == 0:
        return self.evaluate(board)
    
    scores = []
    
    for [(x1, y1), (x2, y2)] in list(board.legal_moves()):
        board.move(x1, y1, x2, y2)

        board.swap_turn_on_current()

        score = self.minmax(remaining_depth - 1, board)
        scores.append(score)

        board.swap_turn_on_current()

        board.undo()
    
    ismaximising = self.board.turn.color != 'white'
    return max(scores) if ismaximising else min(scores)
```

I also opted to profile just one run of `Computer.find_best_move`.

Here is the time profile:

```bash
348027 function calls (347627 primitive calls) in 0.370 seconds

   ncalls        tottime        percall        cumtime        percall filename:lineno(function)
        1    0.000222399    0.000222399    0.369350184    0.369350184 main.py:119(find_best_move)
   420/20    0.001239939    0.000002952    0.349534105    0.017476705 main.py:66(minmax)
      441    0.003106129    0.000007043    0.345628456    0.000783738 main.py:470(legal_moves)
      420    0.013027923    0.000031019    0.334567731    0.000796590 main.py:484(isking_vulnerable)
    10163    0.010648723    0.000001048    0.118487185    0.000011659 main.py:457(legal_nocheck)
      442    0.002617106    0.000005921    0.116887481    0.000264451 main.py:158(__init__)
      440    0.000507580    0.000001154    0.116818532    0.000265497 main.py:230(swap_turn)
    11948    0.010680814    0.000000894    0.065113193    0.000005450 main.py:63(any)
     5345    0.018482915    0.000003458    0.061686059    0.000011541 main.py:420(piece_legal_nocheck)
      442    0.036125347    0.000081732    0.058665112    0.000132726 main.py:250(update_piece_bitboard_data)
```

I felt as though optimising `legal_nocheck` would have a large positive impact on a number of other functions.

Here is a reminder of the function:

```py
def legal_nocheck(self) -> Iterable[list[Coordinate, Coordinate]]:
        """Get all legal moves without checking for checks.
        Returns:
            Iterable(list[Coordinate, Coordinate]): An iterable of pairs of coordinates describing the movement.
        """

        for piece in (p for p in self.self_pieces if p.bb.any()):
            if isinstance(piece, Pawn):
                for x, y in piece.bb.pos():
                    yield from self.get_pawn_movement(x, y)
            else:
                yield from self.piece_legal_nocheck(piece)
```

And here is a shortened time profile for the function:

```bash
   ncalls        tottime        percall        cumtime        percall filename:lineno(function)
       21    0.000024955    0.000001188    0.000298949    0.000014236 main.py:457(legal_nocheck)
        9    0.000036134    0.000004015    0.000118486    0.000013165 main.py:420(piece_legal_nocheck)
        6    0.000031808    0.000005301    0.000108209    0.000018035 main.py:67(pos)
        6    0.000017639    0.000002940    0.000075671    0.000012612 numeric.py:591(argwhere)
        7    0.000004284    0.000000612    0.000072657    0.000010380 main.py:463(<genexpr>)
        6    0.000011595    0.000001933    0.000068373    0.000011396 main.py:63(any)
```

I assessed `Board.isking_vulernable`.

```py
"""In bpard.main:Board:"""

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
```

I wondered whether it was possible to test each move without creating a new instance of a board. I traded `swap_turn` for `swap_turn_on_current` and updated accordingly:

```py
def isking_vulnerable(self) -> bool:
    """Return whether the turn player's king can be taken.
    Returns:
        bool: True if the king can be taken, otherwise False.
    """

    # obtain the legal moves for the other player
    self.swap_turn_on_current()
    other_legals_nocheck = self.legal_nocheck()

    for [(x1, y1), (x2, y2)] in other_legals_nocheck:
        self.move(x1, y1, x2, y2)
        king = self.other_player[self.turn.color].king # obtain the CURRENT player's king

        # if the king is not present, return True
        if not king.bb.any():
            self.undo()
            self.swap_turn_on_current()
            return True
        
        self.undo()
    self.swap_turn_on_current()
    return False
```

However, there were issues with undoing the move.
I decided to update the undo system to work with a list of moves to undo, then when undoing, pop the last index move.

```py
"""In board.main:Board:"""

def __init__(self, ...):
    ...
    # join IDS to pieces
    self.id: dict[int, Piece | Pawn] = {piece.id : piece for piece in self.pieces}

    self.last_pieces_taken: list[Piece | Pawn] = []
    self.last_pieces_moved: list[Piece | Pawn] = []
    self.last_moves: list[list[Coordinate, Coordinate]] = []

    if self.turn.color == 'white':
        ...

def move(self, ...):
    ...
    # update last move data
    self.last_pieces_taken.append(end_piece)
    self.last_pieces_moved.append(start_piece)
    self.last_moves.append(((x1, y1), (x2, y2)))
    ...

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

```

After this change, there were no issues I could notice with the changed functions. However, `Board.swap_turn_on_current` appeared to have no effect.

```py
def swap_turn_on_current(self) -> None:
    """Like Board.swap_turn, but edits the current instance instead of returning a new one."""
    swapped_turn = self.other_player[self.turn.color]
    self.turn = swapped_turn
```

I then realised that `Board.self_bb` and `Board.other_bb` were not being swapped, as well as a number of turn-specific variables. I fixed this:

```py
def swap_turn_on_current(self) -> None:
    """Like Board.swap_turn, but edits the current instance instead of returning a new one."""
    swapped_turn = self.other_player[self.turn.color]
    self.turn = swapped_turn
    
    self.self_bb, self.other_bb = self.other_bb, self.self_bb
    self.self_pieces, self.other_pieces = self.other_pieces, self.self_pieces
    self.self_pos, self.other_pos = self.other_pos, self.self_pos
```

Now there were no issues.

Here is a new time profile for `Computer.find_best_move` since the changes, showing the top 10 slowest functions:

```bash
338293 function calls (337893 primitive calls) in 0.246 seconds

   ncalls        tottime        percall        cumtime        percall filename:lineno(function)
        1    0.000190106    0.000190106    0.245916933    0.245916933 main.py:119(find_best_move)
   420/20    0.001124814    0.000002678    0.230513894    0.011525695 main.py:66(minmax)
      441    0.000731566    0.000001659    0.223756161    0.000507384 main.py:508(legal_moves)
      420    0.011687147    0.000027827    0.215188974    0.000512355 main.py:544(isking_vulnerable)
    10161    0.007119189    0.000000701    0.100766757    0.000009917 main.py:495(legal_nocheck)
    11946    0.009695540    0.000000812    0.058517733    0.000004899 main.py:63(any)
     5349    0.016453074    0.000003076    0.053936194    0.000010083 main.py:426(piece_legal_nocheck)
    11946    0.008544704    0.000000715    0.047221919    0.000003953 fromnumeric.py:2477(any)
     2646    0.012895316    0.000004874    0.040046255    0.000015135 main.py:67(pos)
```

Compared to the time profile before the optimisation:

```bash
348027 function calls (347627 primitive calls) in 0.370 seconds

   ncalls        tottime        percall        cumtime        percall filename:lineno(function)
        1    0.000222399    0.000222399    0.369350184    0.369350184 main.py:119(find_best_move)
   420/20    0.001239939    0.000002952    0.349534105    0.017476705 main.py:66(minmax)
      441    0.003106129    0.000007043    0.345628456    0.000783738 main.py:470(legal_moves)
      420    0.013027923    0.000031019    0.334567731    0.000796590 main.py:484(isking_vulnerable)
    10163    0.010648723    0.000001048    0.118487185    0.000011659 main.py:457(legal_nocheck)
      442    0.002617106    0.000005921    0.116887481    0.000264451 main.py:158(__init__)
      440    0.000507580    0.000001154    0.116818532    0.000265497 main.py:230(swap_turn)
    11948    0.010680814    0.000000894    0.065113193    0.000005450 main.py:63(any)
     5345    0.018482915    0.000003458    0.061686059    0.000011541 main.py:420(piece_legal_nocheck)
      442    0.036125347    0.000081732    0.058665112    0.000132726 main.py:250(update_piece_bitboard_data)
```

Additionally, following this change, `Board.swap_turn` was obsolete. I renamed it to `swap_turn_with_new_instance` and `Board.swap_turn_on_current` to `swap_turn`.

Finally, I updated `Computer.find_best_move` to instead of creating a new `Board` instance, create a single instance at the start of the iteration.

```py
def find_best_move(self, depth: int) -> tuple[int,int,int,int]:
    """Determine the best move based on the evaluation and minmaxing.
    Args:
        depth (int): The depth of the movetree to search down.
    Returns:
        tuple(int, int, int, int): The four indices (x1, y1, x2, y2) pointing to the best move.
    """

    ismaximising = self.board.turn.color == 'white'
    best_move = None
    best_score = -float('inf') if ismaximising else float('inf')
    
    board = self.board.swap_turn_with_new_instance()

    for [(x1, y1), (x2, y2)] in list(board.legal_moves()):
        board.move(x1, y1, x2, y2)
        score = self.minmax(depth - 1, board)

        if (ismaximising and score > best_score) or (not ismaximising and score < best_score):
            best_score = score
            best_move = (x1, y1, x2, y2)
        
        board.undo()
    
    return best_move
```

### `board:Bitboard.any`

Here is the time profile for the 10 slowest functions in `Computer.find_best_move`:

```bash
   13865941 function calls (13850707 primitive calls) in 10.205 seconds

   Ordered by: cumulative time

   ncalls        tottime        percall        cumtime        percall filename:lineno(function)
       25    0.001606387    0.000064255   10.199682586    0.407987303 main.py:92(find_best_move)
15828/594    0.042672281    0.000002696    9.813261274    0.016520642 main.py:41(minmax)
    16447    0.026943646    0.000001638    9.523734686    0.000579056 main.py:509(legal_moves)
    15906    0.533653450    0.000033550    9.220708974    0.000579700 main.py:523(isking_vulnerable)
   434378    0.294382007    0.000000678    4.131147670    0.000009510 main.py:496(legal_nocheck)
   500742    0.409794018    0.000000818    2.452320602    0.000004897 main.py:63(any)
   342773    0.918125287    0.000002679    2.419137617    0.000007058 main.py:427(piece_legal_nocheck)
   500742    0.342666608    0.000000684    1.974030820    0.000003942 fromnumeric.py:2477(any)
   500742    0.572720534    0.000001144    1.631364212    0.000003258 fromnumeric.py:89(_wrapreduction_any_all)
   433706    1.133523286    0.000002614    1.617125266    0.000003729 main.py:310(move)
```

The major call of this `Bitboard.any` was in `Board.legal_nocheck`:

```py
def legal_nocheck(self) -> Iterable[list[Coordinate, Coordinate]]:
    """Get all legal moves without checking for checks.
    Returns:
        Iterable(list[Coordinate, Coordinate]): An iterable of pairs of coordinates describing the movement.
    """

    for piece in (p for p in self.self_pieces if p.bb.any()):
        if isinstance(piece, Pawn):
            for x, y in piece.bb.pos():
                yield from self.get_pawn_movement(x, y)
        else:
            yield from self.piece_legal_nocheck(piece)
```

I decided to keep track of the present pieces, only calculating them in `Board.__init__`.

```py
"""In Board:"""

def __init__(self, ...) -> ...:
    ...
    self.last_pieces_taken: list[Piece | Pawn] = []
    self.last_pieces_moved: list[Piece | Pawn] = []
    self.last_moves: list[tuple[Coordinate, Coordinate]] = []

    self.present_pieces = set(piece.id for piece in self.pieces if piece.bb.any())

    if self.turn.color == 'white':
        self.self_pieces = self.white.pieces
        self.other_pieces = self.black.pieces
    ...

def move(self, ...) -> ...:
    ...
    start_piece.bb[x2, y2] = 1
    # on end bb: set to 0 at end
    end_piece.bb[x2, y2] = 0

    # remove from present_pieces if that was the last piece
    if not end_piece.bb.any() and end_piece.id in self.present_pieces:
        self.present_pieces.remove(end_piece.id)

    # update board
    self.board[x1, y1] = 0
    self.board[x2, y2] = start_id
    ...

def undo(self) -> ...:
  ...
    last_piece_moved.bb[x1, y1] = 1
    self.board[x1, y1] = last_piece_moved.id

    self.present_pieces.add(last_piece_moved.id)

```

Updating accordingly:

```py
"""In Board:"""

def legal_nocheck(self) -> ...:

    for piece in (self.id[pid] for pid in self.present_pieces if self.id[pid] in self.self_pieces):
        if isinstance(piece, Pawn):
            for x, y in piece.bb.pos():
                  ...
```

After a lot of debugging, here is an extract of the time profile:

```bash
15466251 function calls (15451372 primitive calls) in 11.747 seconds

   ncalls         tottime       percall        cumtime        percall filename:lineno(function)
        1    0.000101548    0.000101548   11.747105142   11.747105142 {built-in method builtins.exec}
        1    0.000021499    0.000021499   11.747003594   11.747003594 <string>:1(<module>)
        1    0.000469397    0.000469397   11.746982095   11.746982095 main.py:128(main)
       25    0.001779511    0.000071180   11.741025345    0.469641014 main.py:92(find_best_move)
15484/605    0.043130567    0.000002785   11.244668158    0.018586228 main.py:41(minmax)
    16114    0.027867178    0.000001729   10.971163613    0.000680847 main.py:516(legal_moves)
    15563    0.537365398    0.000034528   10.594932304    0.000680777 main.py:534(isking_vulnerable)
   792151    0.639851213    0.000000808    3.833590054    0.000004839 main.py:63(any)
   411436    1.274354353    0.000003097    3.799443252    0.000009235 main.py:312(move)
   412120    0.312854916    0.000000759    3.514816814    0.000008529 main.py:501(legal_nocheck)
   792151    0.567845453    0.000000717    3.083807811    0.000003893 fromnumeric.py:2477(any)
   792151    0.903698301    0.000001141    2.515962358    0.000003176 fromnumeric.py:89(_wrapreduction_any_all)
   303818    0.855790442    0.000002817    2.351340859    0.000007739 main.py:432(piece_legal_nocheck)
   941022    1.684046111    0.000001790    1.684046111    0.000001790 {method 'reduce' of 'numpy.ufunc' objects}
    96763    0.498165043    0.000005148    1.562602620    0.000016149 main.py:67(pos)
   411411    0.955425311    0.000002322    1.520242439    0.000003695 main.py:365(undo)
```

Unfortunately, the time overall increased significantly. I undid the changes.

I removed the noise on the evaluation so results no longer varied massively.

### `Computer.minmax`

#### Alpha-Beta Pruning

A popular technique called **Alpha-Beta Pruning** could be implemented to reduce the move tree that needs to be searched.

Here is the new function:

```py
def minmax(self, remaining_depth: int, board: Board, alpha: float = -float('inf'), beta: float = float('inf')) -> float:
    """Use minmax with alpha-beta pruning to score a position.
    Args:
        remaining_depth (int): The remaining depth of the search.
        board (Board): The board state to apply minmax to.
        alpha (float): The alpha variable for Alpha-Beta Pruning.
        beta (float): The beta variable for Alpha-Beta Pruning.
    Returns:
        float: The score.
    """

    if remaining_depth == 0:
        return self.evaluate(board)
    
    scores = []
    
    for [(x1, y1), (x2, y2)] in list(board.legal_moves()):
        board.move(x1, y1, x2, y2)
        board.swap_turn()

        score = self.minmax(remaining_depth - 1, board, alpha, beta)
        scores.append(score)

        board.swap_turn()
        board.undo()

        # Alpha-Beta pruning
        ismaximising = self.board.turn.color != 'white'
        if ismaximising:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
        
        if beta <= alpha:
            break  # Prune the remaining branches
    
    return max(scores) if ismaximising else min(scores)
```

I won't show the time profile for this function; there won't be evidence of change at this low a depth.
