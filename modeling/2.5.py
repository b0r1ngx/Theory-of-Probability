import random as r

# 0 is a whitespaces cell's

answer = 23 / 240
attempts = 500000
checkmates = 0
# fill the board
for i in range(attempts):
    board_3x8 = [['0' for j in range(8)] for i in range(3)]
    column_black_king = r.randint(0, 7)
    board_3x8[2][column_black_king] = 'B'
    column_for_king = r.randint(0, 7)
    board_3x8[0][column_for_king] = 'W'
    if column_for_king == 0:
        board_3x8[0][column_for_king + 1] = '1'
        board_3x8[1][column_for_king] = '1'
        board_3x8[1][column_for_king + 1] = '1'
    elif column_for_king == 7:
        board_3x8[0][column_for_king - 1] = '1'
        board_3x8[1][column_for_king] = '1'
        board_3x8[1][column_for_king - 1] = '1'
    else:
        board_3x8[0][column_for_king - 1] = '1'
        board_3x8[0][column_for_king + 1] = '1'
        board_3x8[1][column_for_king + 1] = '1'
        board_3x8[1][column_for_king] = '1'
        board_3x8[1][column_for_king - 1] = '1'
    F = 0
    list_of = []
    for it in range(1, 3):
        for j in range(8):
            if board_3x8[it][j] != 'B':
                list_of.append([it, j])
    queen_coord = r.choice(list_of)
    board_3x8[queen_coord[0]][queen_coord[1]] = 'F'
    for it in range(8):
        if board_3x8[queen_coord[0]][it] == '0':
            board_3x8[queen_coord[0]][it] = '1'

    if board_3x8[0][queen_coord[1]] == '0':
        board_3x8[0][queen_coord[1]] = '1'
    if board_3x8[1][queen_coord[1]] == '0':
        board_3x8[1][queen_coord[1]] = '1'
    if board_3x8[2][queen_coord[1]] == '0':
        board_3x8[2][queen_coord[1]] = '1'

    if board_3x8[1][queen_coord[1]] == 'F':
        if queen_coord[1] == 0:
            if board_3x8[2][queen_coord[1]] != 'B':
                board_3x8[2][queen_coord[1]] = '1'
            if board_3x8[2][queen_coord[1] + 1] != 'B':
                board_3x8[2][queen_coord[1] + 1] = '1'
        elif queen_coord[1] == 7:
            if board_3x8[2][queen_coord[1] - 1] != 'B':
                board_3x8[2][queen_coord[1] - 1] = '1'
            if board_3x8[2][queen_coord[1]] != 'B':
                board_3x8[2][queen_coord[1]] = '1'
        else:
            if board_3x8[2][queen_coord[1] - 1] != 'B':
                board_3x8[2][queen_coord[1] - 1] = '1'
            if board_3x8[2][queen_coord[1]] != 'B':
                board_3x8[2][queen_coord[1]] = '1'
            if board_3x8[2][queen_coord[1] + 1] != 'B':
                board_3x8[2][queen_coord[1] + 1] = '1'
    else:
        if queen_coord[1] == 0:
            board_3x8[1][queen_coord[1]] = '1'
            board_3x8[1][queen_coord[1] + 1] = '1'
        elif queen_coord[1] == 7:
            board_3x8[1][queen_coord[1] - 1] = '1'
            board_3x8[1][queen_coord[1]] = '1'
        else:
            board_3x8[1][queen_coord[1] - 1] = '1'
            board_3x8[1][queen_coord[1]] = '1'
            board_3x8[1][queen_coord[1] + 1] = '1'
    print(queen_coord)

    if queen_coord == [1, 0] or queen_coord == [2, 0]:
        if board_3x8[queen_coord[0] - 1][queen_coord[1]] != 'W' \
                and board_3x8[queen_coord[0] - 1][queen_coord[1] + 1] != 'W':
            board_3x8[queen_coord[0]][queen_coord[1]] = '0'
    elif queen_coord == [1, 7] or queen_coord == [2, 7]:
        if board_3x8[queen_coord[0] - 1][queen_coord[1] - 1] != 'W' \
                and board_3x8[queen_coord[0] - 1][queen_coord[1]] != 'W':
            board_3x8[queen_coord[0]][queen_coord[1]] = '0'
    else:
        if board_3x8[queen_coord[0] - 1][queen_coord[1] - 1] != 'W' \
                and board_3x8[queen_coord[0] - 1][queen_coord[1]] != 'W' \
                and board_3x8[queen_coord[0] - 1][queen_coord[1] + 1] != 'W':
            board_3x8[queen_coord[0]][queen_coord[1]] = '0'

    if column_black_king == 0:
        if board_3x8[1][column_black_king] != '0' \
                and board_3x8[1][column_black_king + 1] != '0' \
                and board_3x8[2][column_black_king + 1] != '0':
            checkmates += 1
    elif column_black_king == 7:
        if board_3x8[1][column_black_king - 1] != '0' \
                and board_3x8[1][column_black_king] != '0' \
                and board_3x8[2][column_black_king - 1] != '0':
            checkmates += 1
    else:
        if board_3x8[1][column_black_king - 1] != '0' \
                and board_3x8[1][column_black_king] != '0' \
                and board_3x8[1][column_black_king + 1] != '0' \
                and board_3x8[2][column_black_king - 1] != '0' \
                and board_3x8[2][column_black_king + 1] != '0':
            checkmates += 1

    for it in range(3):
        print(board_3x8[it])
    print(checkmates)
    print()
print('Analytic answer:', answer)
print('Laboratory:', checkmates / attempts)
