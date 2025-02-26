# default modules
import os

from board.main import Board

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
        self.board = board

    def recursive_search()


###################################################################################################


# Post-class Functions


###################################################################################################


def main() -> None:
    """The main program."""

    board: Board = Board().default()
    computer = Computer(board)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nExecution interrupted.")