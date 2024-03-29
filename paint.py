import pygame

from pygame.locals import *

pygame.init()

extra_spaceY = 150
extra_spaceX = 0

mult = 15

width = 28 * mult + extra_spaceX
height = 28 * mult + extra_spaceY
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("SquareDraw")

# Grid Creator
numberOfRows = 28
numberOfColumns = 28
# grid = Grid(numberOfRows, numberOfColumns)
grid = [
    [255 for x in range(numberOfRows)] for y in range(numberOfColumns)
]  # use array for grid: 0=white, 1=black

# Medidas
basicX = (width - extra_spaceX) / numberOfColumns
basicY = (height - extra_spaceY) / numberOfRows


def drawScreen(screen, grid):  # draw rectangles from grid array
    for i in range(numberOfColumns):
        for j in range(numberOfRows):
            if grid[i][j]:
                # print('yes')
                # print(i, j)
                # print(grid[i][j])
                grid[i][j] = max(grid[i][j], 1)
                pygame.draw.rect(
                    screen,
                    (grid[i][j], grid[i][j], grid[i][j]),
                    (j * basicX, i * basicY, basicX, basicY),
                )


screen.fill((255, 255, 255))  # start screen
import pygame
from digits import read_grid

font = pygame.font.SysFont("arial", 32)


clear_text = "Clear"
clear_text = font.render(clear_text, True, (0, 0, 0))
# Create a pygame.Rect object that represents the button's boundaries
clear_rect = clear_text.get_rect()
clear_rect.center = (width // 2 - 100, height - 100)


Number = ""


def update():
    screen.fill((255, 255, 255))
    screen.blit(clear_text, clear_rect)
    number_text = font.render(Number, True, (0, 0, 0))
    number_rect = number_text.get_rect()
    number_rect.center = (width // 2 + 100, height - 100)

    screen.blit(number_text, number_rect)
    drawScreen(screen, grid)
    pygame.display.flip()


def clearScreen():
    global grid
    screen.fill((255, 255, 255))
    grid = [[255 for x in range(numberOfRows)] for y in range(numberOfColumns)]


right_clicking = 0
left_clicking = 0
import matplotlib.pyplot as plt

while True:
    events = pygame.event.get()
    for event in events:
        if (event.type == pygame.QUIT) or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            pygame.quit()

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and clear_rect.collidepoint(event.pos)
        ):
            clearScreen()

        if (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == 3
        ) or right_clicking:  # mouse button down
            right_clicking = 1
            x, y = pygame.mouse.get_pos()

            xInGrid = int(x / basicX)
            yInGrid = int(y / basicY)
            try:
                grid[yInGrid][xInGrid] = 255
            except IndexError:
                continue

            screen.fill((255, 255, 255))
            drawScreen(screen, grid)
        if (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
        ) or left_clicking:  # mouse button down
            print("left")
            left_clicking = 1
            x, y = pygame.mouse.get_pos()

            xInGrid = int(x / basicX)
            yInGrid = int(y / basicY)
            try:
                grid[yInGrid][xInGrid] -= 25
                grid[yInGrid][xInGrid + 1] -= 6
                grid[yInGrid][xInGrid - 1] -= 6
                grid[yInGrid + 1][xInGrid] -= 6
                grid[yInGrid - 1][xInGrid] -= 6
            except IndexError:
                left_clicking = 0
                continue

            # pygame.draw.rect(
            #     screen, (0, 0, 0), (xInGrid * basicX, yInGrid * basicY, basicX, basicY)
            # )

        if event.type == pygame.MOUSEBUTTONUP:
            left_clicking = 0
            right_clicking = 0

            Number = str(read_grid(grid))

        update()
