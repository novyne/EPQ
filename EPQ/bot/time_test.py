import cProfile
import os
import pstats
import time

import main as main

#############################################################################################################################

PROFILE_FILE = 'cprofile.prof'

def f8_alt(x):
    return "%14.9f" % x
pstats.f8 = f8_alt

#############################################################################################################################

def warmup():
    start = time.perf_counter()
    for _ in range(int(1e6)):
        _ = 12345 ** 100

    print(f"Warmup complete in {time.perf_counter() - start:.2f} seconds.")

warmup()

#############################################################################################################################

cProfile.run('main.main()',PROFILE_FILE,sort='cumulative')

stats = pstats.Stats(PROFILE_FILE)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats()

#############################################################################################################################

os.remove(PROFILE_FILE)