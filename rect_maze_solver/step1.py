import numpy as np
import cv2

np.set_printoptions(threshold=np.nan)


def readImage(filePath):
    img = cv2.imread(filePath, 0)
    ret, binaryimg = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return binaryimg


def findNeighbours(img, row, col):
    neigh = []
    i = 0
    while (True):
        if img[i, i] != 0:
            border = i
            break
        i += 1
    i = border
    cell = 0
    while (True):
        if img[i, i] == 0:
            cell = i
            break
        i += 1
    if img[cell - border * 2, cell] == 0 or img[cell, cell - border * 2] == 0:
        cell += border
    row_new = (2 * row + 1) * (cell / 2)
    col_new = (2 * col + 1) * (cell / 2)
    top = row_new - (cell / 2) + 1
    bottom = row_new + (cell / 2) - 2
    left = col_new - (cell / 2) + 1
    right = col_new + (cell / 2) - 2
    if img[top, col_new] != 0:
        neigh.append([row - 1, col])
    if img[bottom, col_new] != 0:
        neigh.append([row + 1, col])
    if img[row_new, left] != 0:
        neigh.append([row, col - 1])
    if img[row_new, right] != 0:
        neigh.append([row, col + 1])
    return neigh


def colorCell(img, row, column, colorVal):
    neighbours = findNeighbours(img, row, column)
    i = 0
    while (True):
        if img[i, i] != 0:
            border = i
            break
        i += 1
    i = border
    cell = 0
    while (True):
        if img[i, i] == 0:
            cell = i
            break
        i += 1
    if img[cell - border * 2, cell] == 0 or img[cell, cell - border * 2] == 0:
        cell = cell + border
    row_new = (2 * row + 1) * (cell / 2)
    col_new = (2 * column + 1) * (cell / 2)
    top = row_new - (cell / 2) + 1
    bottom = row_new + (cell / 2) - 2
    left = col_new - (cell / 2) + 1
    right = col_new + (cell / 2) - 2
    for i in neighbours:
        if i[0] == row + 1:
            img[top + 1: bottom + 2, left + 1: right] = colorVal
        if i[0] == row - 1:
            img[top - 1: bottom, left + 1: right] = colorVal
        if i[1] == column + 1:
            img[top + 1: bottom, left + 1: right + 2] = colorVal
        if i[1] == column - 1:
            img[top + 1: bottom, left - 1: right] = colorVal

    ###################################################
    return img


def main(filePath):
    img = readImage(filePath)
    coords = [(0, 0), (9, 9), (3, 2), (4, 7), (8, 6)]
    string = ""
    for coord in coords:
        img = colorCell(img, coord[0], coord[1], 150)
        neigh = findNeighbours(img, coord[0], coord[1])
        print(neigh)
        string += str(neigh) + "\n"
        for k in neigh:
            img = colorCell(img, k[0], k[1], 230)
        if __name__ == '__main__':
            return img
        else:
            return string + "\t"


if __name__ == '__main__':
    filePath = 'maze00.jpg'
    img = main(filePath)
    cv2.imshow('canvas', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
