from PIL import Image

from gridworld.GridWorld import GridWorld, Coordinate

TILE_SIZE = 7

ARROW_UP = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

ARROW_DOWN = [ARROW_UP[index] for index in range(len(ARROW_UP) - 1, -1, -1)]
ARROW_LEFT = [*zip(*ARROW_UP)]
ARROW_RIGHT = [[ARROW_LEFT[row][col] for col in range(len(ARROW_UP) - 1, -1, -1)] for row in range(len(ARROW_UP))]


def draw_world(world: GridWorld, color_mapping):
    image = Image.new(mode="RGB", size=(world.width * TILE_SIZE, world.height * TILE_SIZE), color=0x000000)

    for x in range(world.width):
        for y in range(world.height):
            fill_tile(image, world, x, y, color_mapping(world.get(Coordinate(x, y))))

    return image


def fill_tile(image, world, x, y, color):
    # convert coordinate so that bottom left is (0, 0)
    # PIL: top left is (0, 0)
    y = world.height - y - 1

    for x_offset in range(TILE_SIZE):
        for y_offset in range(TILE_SIZE):
            image.putpixel((x * TILE_SIZE + x_offset, y * TILE_SIZE + y_offset), color)


def draw_shape(image, world, x, y, shape, color):
    # convert coordinate so that bottom left is (0, 0)
    # PIL: top left is (0, 0)
    y = world.height - y - 1

    for x_offset in range(TILE_SIZE):
        for y_offset in range(TILE_SIZE):
            if shape[y_offset][x_offset]:
                image.putpixel((x * TILE_SIZE + x_offset, y * TILE_SIZE + y_offset), color)