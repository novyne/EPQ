import cProfile
import main

cProfile.run('list(main.Board().legal_moves())',sort='cumulative')