# This is necessary to find the main code
import sys
import math

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back, Style, init
init(autoreset=True)

class TestCharacter(CharacterEntity):
    def do(self, wrld):
        path = self.aStar(wrld, (7, 18))
        move = path[len(path) - 1]
        self.move(move[0] - self.x, move[1] - self.y)
        pass

    def aStar(self, wrld, goal):
        frontier = []
        frontier.append(((self.x, self.y), 0))
        came_from = {}
        cost_so_far = {}
        came_from[(self.x, self.y)] = None
        cost_so_far[(self.x, self.y)] = 0
        move = 1
        print("SELFX " + str(self.x))
        print("SELFY " + str(self.y))

        monsters = []

        # Iterates through board to find monsters and exit85
        for x in range(0, wrld.width()):
            for y in range(0, wrld.height()):
                if wrld.monsters_at(x, y):  # Finds all the monsters in the board
                    monsters.append((x, y))

        while not len(frontier) == 0:
            frontier.sort(key=lambda tup: tup[1])  # check that
            current = frontier.pop(0)
            self.set_cell_color(current[0][0], current[0][1], Fore.RESET + Back.RESET)  # resets color
            if (current[0][0], current[0][1]) == goal:
                break
            for next in get_adjacent(current[0], wrld):
                if wrld.wall_at(next[0], next[1]):
                    cost_so_far[(next[0], next[1])] = 999
                    new_cost = 1000
                elif (next[0], next[1]) in monsters:
                    cost_so_far[(next[0], next[1])] = 99
                    new_cost = 100
                else:
                    new_cost = self.cost_to(current[0], next) + cost_so_far[current[0]]
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    frontier.append((next, new_cost + self.manhattan_distance(next[0], next[1], goal[0], goal[1])))
                    came_from[next] = current[0]

        cursor = goal

        path = []
        while not cursor == (self.x, self.y):
            self.set_cell_color(cursor[0], cursor[1], Fore.RED + Back.GREEN)
            move = cursor
            path.append(cursor)
            try:
                cursor = came_from[cursor]
            except KeyError:
                self.move(0, 0)
                pass
                break
        print("PATH: ")
        print(path)

        return path

    def manhattan_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    # Prioritizes downward
    def cost_to(self, current, next):
        diff = (next[0] - current[0], next[1] - current[1])
        val = abs(diff[0]) + abs(diff[1])
        if val == 2:
            return 2
        else:
            return 1

    # Resets styling for each cell. Prevents unexpected/inconsistent behavior that otherwise appears with coloring.
    def reset_cells(self, wrld):
        for x in range(0, wrld.width()):
            for y in range(0, wrld.height()):
                self.set_cell_color(x, y, Fore.RESET + Back.RESET)

    def printOurWorld(self, wrld, cost_so_far):
        w, h = len(wrld.grid), len(wrld.grid[0])
        print('\n\n')
        world = [[0 for x in range(w)] for y in range(h)]

        # for row in world:
        #     print(row)
        keys = cost_so_far.keys()

        for key in keys:
            # x,y = val[0][0], val[0][1]
            # print(x, y)
            # print("Set value at " + str(val[0][0]-1) + " , " + str(val[0][1] -1) + " to: " + str(val[1]) )

            k = str(key).split(',')

            world[key[1]][key[0]] = "{0:03d}".format(cost_so_far[key])
        world[self.y][self.x] = "X"

        for row in world:
            print(row)

    # Returns a list of tiles which are unsafe due to monsters.
    def monster_tiles(self, wrld):
        tiles = []
        for x in range(0, wrld.width()):
            for y in range(0, wrld.height()):
                if wrld.monsters_at(x, y):
                    for t in get_adjacent((x, y), wrld):
                         tiles.append(t)
                    tiles.append((x, y))
        print("Monster tiles: ")
        print(tiles)
        return tiles

    # ==================== FEATURES ==================== #
    #   - Distance to closest bomb
    #   - Distance to closest monster
    #   - 1 / (Distance to exit)^2

    # Returns an integer representing the Manhattan distance to the closest bomb.
    def closest_bomb(self):
        return 1

    # Returns an integer representing the Manhattan distance to the closest monster.
    def closest_monster(self):
        return 1

    # Returns the 1/(A* distance to exit)^2.
    def distance_to_exit(self):
        return 1

    def aStar(self, wrld, mapTo, toExit=True):
        self.reset_cells(wrld)
        # A*
        frontier = []
        frontier.append(((self.x, self.y), 0))
        came_from = {}
        cost_so_far = {}
        came_from[(self.x, self.y)] = None
        cost_so_far[(self.x, self.y)] = 0
        move = 1
        print("SELFX " + str(self.x))
        print("SELFY " + str(self.y))

        monsters = []
        ex = (7, 18)
        if toExit:
            # Iterates through board to find monsters and exit
            for x in range(0, wrld.width()):
                for y in range(0, wrld.height()):
                    if wrld.monsters_at(x, y):  # Finds all the monsters in the board
                        monsters.append((x, y))
                    if wrld.exit_at(x, y):  # Just in case exit is not where we expect it to be in the bottom right corner
                        ex = (x, y)

            for t in monsters:
                self.set_cell_color(t[0], t[1], Fore.RED + Back.RED)

        while not len(frontier) == 0:
            frontier.sort(key=lambda tup: tup[1])  # check that
            current = frontier.pop(0)
            if toExit:
                self.set_cell_color(current[0][0], current[0][1], Fore.RESET + Back.RESET)  # resets color
            if (current[0][0], current[0][1]) == ex:
                break
            for next in get_adjacent(current[0], wrld):
                if wrld.wall_at(next[0], next[1]):
                    cost_so_far[(next[0], next[1])] = 999
                    new_cost = 1000
                elif (next[0], next[1]) in monsters:
                    cost_so_far[(next[0], next[1])] = 99
                    new_cost = 100
                else:
                    new_cost = self.cost_to(current[0], next) + cost_so_far[current[0]]
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    frontier.append((next, new_cost + self.manhattan_distance(next[0], next[1], ex[0], ex[1])))
                    came_from[next] = current[0]

        # self.printOurWorld(wrld, cost_so_far)

        cursor = ex
        path = []
        while not cursor == (self.x, self.y):
            if toExit:
                self.set_cell_color(cursor[0], cursor[1], Fore.RED + Back.GREEN)
            move = cursor
            path.append(cursor)
            try:
                cursor = came_from[cursor]
            except KeyError:
                self.move(0, 0)
                pass
                break
        print("PATH: ")
        print(path)

        if not len(path) == 0:
            move = path[len(path) - 1]

        # carries momentum? mayhaps not the best
        return move[0] - self.x, move[1] - self.y, len(path)
        # self.move(move[0] - self.x, move[1] - self.y)

# Returns a list of coordinates in the world surrounding the current one.
# param current: An (x, y) point
def get_adjacent(current, wrld):
    # Returns a list of points in the form [(x1, y1), (x2, y2)]
    neighbors = []
    x = current[0]
    y = current[1]

    if x >= 1:
        if y >= 1:
            neighbors.append((x - 1, y - 1))  # top left
        neighbors.append((x - 1, y))  # middle left
        if y < wrld.height() - 1:
            neighbors.append((x - 1, y + 1))  # bottom left

    if y >= 1:
        neighbors.append((x, y - 1))  # top middle
    if y < wrld.height() - 1:
        neighbors.append((x, y + 1))  # bottom middle

    if x < wrld.width() - 1:
        if y >= 1:
            neighbors.append((x + 1, y - 1))  # top right
        neighbors.append((x + 1, y))  # middle right
        if y < wrld.height() - 1:
            neighbors.append((x + 1, y + 1))  # bottom right

    return neighbors

def printFrontier(frontier):
    for val in frontier:
        print(val)


