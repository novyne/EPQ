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


# globals and pre-main functions go here :)


###################################################################################################


# classes go here :)


###################################################################################################


# functions go here :)


###################################################################################################


def main() -> None:
    """The main program."""

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nGoodbye!")