ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_SIZE = len(ALPHABET)

# Field values to parse the csv
LETTER_ID = 0
LETTER_VALUE = 1
NEXT_ID = 2
WORD_ID = 3
POSITION = 4
FOLD = 5
FIRST_PIXEL = 6
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 8
NB_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
NB_FEATURES = ALPHABET_SIZE * (NB_PIXELS + ALPHABET_SIZE + 3)

MAX_LENGTH = 20
