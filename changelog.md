# Pre-Alpha
## PA 0.0.1 - 15/10/2024
- Created the board as well as a legal move checker
    - missing castling and promotion logic
- Created two computers which play random moves against each other
- Made a text-based GUI (the pieces are too small to see so what's the point)

## PA Patch Notes - 27/10/2024
- Created a spritesheet as well as a class to load the sprites into `PIL` Images
- Created a function to display the board
    - currently extremely slow
- Realised the legal move checker is BROKEN!!

## PA Patch Notes - 29/10/2024
- Fixed the legal move checker
- Made the GUI update several times faster by removing unnecessary canvas updates
- Added promotion logic (untested)
- Added check and promotion notation to the SAN function (check notation is broken and promotion notation is untested)
- Added checkmate and stalemate by no legal moves logic
    - Working on full stalemate logic, currently broken
- Added a captured pieces indicator (shown by the smaller pieces at the top of the board)

## PA Patch Notes - 30/10/2024
- Added more stalemate logic
    - some logic is still missing