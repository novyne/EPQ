# EPQ Menu - The Computer

## Framework

To create this computer, I decided to use a typical approach for board game computers - **Minmaxing**. The process is as follows:

* Use a recursive method to 'search' every move `n` moves in the future
* Score the board position with an algorithm
* Apply a *minmaxing* algorithm to decide the move required to maximise / minimise the best possible score.

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
        print(f"\nGoodbye!")
```

Firstly, I created a class, taking just the board as a *superparamater*.
