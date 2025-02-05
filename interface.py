import os
import tkinter as tk

from PIL import Image, ImageTk


class App(tk.Tk):

    """Main application class."""

    CAPTURED_PIECE_POS = {
        1 : [(32, 16), (48, 16), (64, 16), (80, 16), (96, 16), (112, 16), (128, 16), (144, 16)],
        2 : [(160, 16), (176, 16)],
        3 : [(192, 16), (208, 16)],
        4 : [(224, 16), (240, 16)],
        5 : [(256, 16)],

        7 : [(32, 0), (48, 0), (64, 0), (80, 0), (96, 0), (112, 0), (128, 0), (144, 0)],
        8 : [(160, 0), (176, 0)],
        9 : [(192, 0), (208, 0)],
        10 : [(224, 0), (240, 0)],
        11 : [(256, 0)]
    }

    def __init__(self) -> None:
        """Initialize the application."""

        super().__init__()

        self.title('Chess')

        self.cv = tk.Canvas(self, width=640, height=640, bg="white")
        self.cv.pack()
        
        self.s = Sprites("spritesheet.png")
        self.sprites = self.s.sprites

    def draw_board(self, board) -> None:
        """Load the given board.
        Args:
            board (Board): The board to be drawn.
        """

        self.cv.images = [] # clear canvas
        self.s.place(self.cv, "board", 0, 0) # place board sprite

        for x in range(8):
            for y in range(8):
                ID = board.board[x, 7 - y]
                if ID == 0: continue

                self.s.place(self.cv, str(ID), 
                             x * 32 + 32,
                             y * 32 + 32)
                
###################################################################################################


class Sprites:

    """Sprites class."""

    POSITIONS: dict[str, tuple[tuple[int,int],tuple[int,int]]] = {
        "1" : ((0,0), (32,32)),
        "2" : ((32,0), (64,32)),
        "3" : ((64,0), (96,32)),
        "4" : ((96,0), (128,32)),
        "5" : ((128,0), (160,32)),
        "6" : ((160,0), (192,32)),

        "7" : ((0,32), (32,64)),
        "8" : ((32,32), (64,64)),
        "9" : ((64,32), (96,64)),
        "10" : ((96,32), (128,64)),
        "11" : ((128,32), (160,64)),
        "12" : ((160,32), (192,64)),
        
        "s1" : ((192,0), (208,16)),
        "s2" : ((208,0), (224,16)),
        "s3" : ((224,0), (240,16)),
        "s4" : ((240,0), (256,16)),
        "s5" : ((256,0), (272,16)),

        "s7" : ((192,32), (208,48)),
        "s8" : ((208,32), (224,48)),
        "s9" : ((224,32), (240,48)),
        "s10" : ((240,32), (256,48)),
        "s11" : ((256,32), (272,48)),

        "board" : ((0, 64), (320, 384)),
        "availablemovesquare" : ((272, 0), (304, 32))
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