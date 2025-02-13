import cProfile
import main

board = main.Board().default()

cProfile.run('list(board.legal_moves())',sort='cumulative')