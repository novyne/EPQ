# default modules
import cProfile
import os
import random as rn

from EPQ.bot.board.board import Board

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


# Globals & Pre-class Functions


###################################################################################################


class Computer:

    def __init__(self, board: Board) -> None:
        """Computer class. Evaluates the 'best' move from a given Board instance.
        Args:
            board (Board): The board instance to evaluate.
        """

        self.board = board

        self.material = [0] + [1, 3, 3.25, 5, 9, float('inf')] * 2
    
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
            board.swap_turn()

            score = self.minmax(remaining_depth - 1, board)
            scores.append(score)

            board.swap_turn()
            board.undo()
        
        ismaximising = self.board.turn.color != 'white'
        return max(scores) if ismaximising else min(scores)
    
    def evaluate(self, board: Board) -> float:
        """Score a given board position.
        Args:
            board (Board): The board instance to score.
        Returns:
            float: The score.
        """

        score = 0

        for piece in board.white.pieces[:-1]:
            count = piece.bb.bb.sum()
            count *= self.material[piece.id]

            score += count

        for piece in board.black.pieces[:-1]:
            count = piece.bb.bb.sum()
            count *= self.material[piece.id]

            score -= count

        return score

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


###################################################################################################


# Post-class Functions


###################################################################################################


def main() -> None:
    """The main program."""

    board: Board = Board().default()

    for _ in range(25):
        computer = Computer(board)
        x1, y1, x2, y2 = computer.find_best_move(depth=2)
        board.move(x1, y1, x2, y2)
        board.swap_turn()
        print(board)

    # Computer(board).find_best_move(depth=2)


if __name__ == '__main__':
    try:
        cProfile.run('main()',sort='cumulative')
    except KeyboardInterrupt:
        print(f"\nExecution interrupted.")