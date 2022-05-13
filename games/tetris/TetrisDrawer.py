import pygame

from games.tetris.Tetris import Tetris, STATE_RUNNING, STATE_GAME_OVER

TILE_SIZE = 20

COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 128, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_FUCHSIA = (255, 0, 255)
COLOR_LIME = (0, 255, 0)
COLOR_NAVY = (0, 0, 128)

FIGURE_COLORS = [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW, COLOR_FUCHSIA, COLOR_LIME, COLOR_NAVY]


def color_tile(screen, x, y, color):
    pygame.draw.rect(
        screen,
        color,
        [x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE]
    )


def draw_current_figure(screen, game: Tetris):
    figure = game.current_figure

    for y_offset in range(figure.height):
        for x_offset in range(figure.width):
            if figure.is_occupied(x_offset, y_offset):
                color_tile(screen, figure.x + x_offset, figure.y + y_offset, FIGURE_COLORS[figure.color - 1])


if __name__ == "__main__":
    game = Tetris(15, 20)

    screen_width = game.width * TILE_SIZE
    screen_height = game.height * TILE_SIZE

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("TETRIS")

    fps = 5
    clock = pygame.time.Clock()

    running = True

    while running:
        screen.fill(COLOR_WHITE)

        # draw current board state
        for y in range(game.height):
            for x in range(game.width):
                if game.is_occupied(x, y):
                    color_tile(screen, x, y, FIGURE_COLORS[game.get_state(x, y) - 1])

        if game.current_state == STATE_RUNNING:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.rotate_current_figure()
                    elif event.key == pygame.K_a:
                        game.move_left()
                    elif event.key == pygame.K_d:
                        game.move_right()
                    elif event.key == pygame.K_s:
                        game.move_down()

            draw_current_figure(screen, game)
            game.move_down()

        elif game.current_state == STATE_GAME_OVER:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game.reset()

            font = pygame.font.SysFont(None, 50, bold=True)
            text = font.render('GAME OVER', True, COLOR_RED)
            text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(fps)

