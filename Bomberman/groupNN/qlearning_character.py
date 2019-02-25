# This is necessary to find the main code
import sys
import operator
import random
import math
import numpy as np

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from entity import AIEntity
from entity import MovableEntity
from colorama import Fore, Back, Style, init
from sensed_world import SensedWorld
from events import Event
init(autoreset=True)

class QCharacter(CharacterEntity):

    def __init__(self, qtable, *args, **kwargs):
        super(QCharacter, self).__init__(*args, **kwargs)
        # Whether this character wants to place a bomb
        self.maybe_place_bomb = False
        # Debugging elements
        self.tiles = {}
        self.qtable = qtable

    def do(self, wrld):
        if not self.threatened(wrld):
            path = aStar((self.x, self.y), wrld, (7, 18))
            print(path)
            move = path[len(path) - 1]
            self.move(move[0] - self.x, move[1] - self.y)
            pass
        else:
            # IF threatened we want to
            # 1st move away from monster
            # THEN move toward goal.
            # QLEARNING FOR THIS???
            state = calculate_state((self.x, self.y), wrld)

            actions = self.valid_moves(wrld)
            for a in actions:
                 self.setQ(state, a, wrld)

            move = self.select_best_move(state, actions, wrld)  #  path[len(path) - 1]

            print(self.x, self.y)
            print(wrld.exitcell)
            if (self.x, self.y) == (wrld.exitcell[0], wrld.exitcell[1]):
                print("\n\n WINNNNN \n\n")

            print("MOVE:")
            print(move)
            self.move(move[0], move[1])
            pass

    def setQ(self, state, action, wrld):
        alpha = 0.8
        gamma = 0.5
        keys = self.qtable.keys()
        ra = (action[0] - self.x, action[1] - self.y)  # Relative action
        if (state, ra) not in keys:
            self.qtable[(state, ra)] = 0

        self.qtable[(state, ra)] = self.qtable[state, ra] + alpha * (reward(self, wrld) + gamma * self.getNextBestScore(state, wrld) - self.qtable[state, ra])

    def getNextBestScore(self, state, wrld):
        actions = get_adjacent((self.x, self.y), wrld)
        keys = self.qtable.keys()
        for a in actions:
            # Check that there isn't a wall at this move
            if not wrld.wall_at(a[0], a[1]):
                # Check the reward we end up at after making this move
                # Initialize state value to 0 if we haven't seen it before
                ra = a[0] - self.x, a[1] - self.y  # RELATIVE action based on our position
                if (state, ra) not in keys:
                    self.qtable[(state, ra)] = 0
                # Simulate taking this action and see what happens
                sim = SensedWorld.from_world(wrld)  # creates simulated world
                c = sim.me(self)  # finds character from simulated world
                c.move(ra[0] - self.x, ra[1] - self.y)  # moves character in simulated world
                s = sim.next()  # updates simulated world
                c = s[0].me(c)  # gives us character. this is a tuple, we want the board, not the list of elapsed events

                # Check if game is over
                if c is None:
                    print("Game can end!")
                    print(s[1][0])
                    for event in s[1]:
                        if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER and event.character.name == self.name:
                            return -100
                        elif event.tpe == Event.CHARACTER_FOUND_EXIT and event.character.name == self.name:
                            return 100
                else:
                    return reward(c, wrld)

    def select_best_move(self, state, moves, wrld):
        candidates = []
        # Construct table keys from possible moves and current state.
        for m in moves:
            if not wrld.wall_at(m[0], m[1]):
                rm = (m[0] - self.x, m[1] - self.y)
                print(rm)  # Relative move #TODO THIS MAY NOT BE RIGHT. Maybe should be m[1] - self.x
                candidates.append((state, rm))

        m = -1000

        moves = []

        for c in candidates:
            print("Move, score;")
            print(c, self.qtable[c])
            if self.qtable[c] > m:
                moves.clear()
                m = self.qtable[c]
                moves.append(c[1])
            elif m == self.qtable[c]:
                moves.append(c[1])

        return random.choice(moves)

    def threatened(self, wrld):
        # TODO Add a check for bombs as well
        if closest_monster((self.x, self.y), wrld) <= 2:
            return True
        return False

    def valid_moves(self, wrld):
        moves = get_adjacent((self.x, self.y), wrld)
        final = []
        for m in moves:
            if not wrld.wall_at(m[0], m[1]):
                final.append(m)
        return final

    # Resets styling for each cell. Prevents unexpected/inconsistent behavior that otherwise appears with coloring.
    def reset_cells(self, wrld):
        for x in range(0, wrld.width()):
            for y in range(0, wrld.height()):
                self.set_cell_color(x, y, Fore.RESET + Back.RESET)

    def printWorld(self, wrld):
        w, h = len(wrld.grid), len(wrld.grid[0])
        print('\n\n')
        world = [[0 for x in range(w)] for y in range(h)]

        world[self.y][self.x] = "X"

        for row in world:
            print(row)

# ==================== STATIC METHODS ==================== #
# Many of our methods actually need to be static because they need to be applied to different world state objects.

# Calculates manhattan distance between the two sets of coordinates. These could be tuples, but whatever.
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


# Prioritizes downward
def cost_to(current, next):
    diff = (next[0] - current[0], next[1] - current[1])
    val = abs(diff[0]) + abs(diff[1])
    if val == 2:
        return 2
    else:
        return 1


# Returns a vector of values representing each feature.
# Vector structure: (bomb distance, monster distance, exit distance)
def calculate_state(coords, wrld):
    monster = closest_monster((coords[0], coords[1]), wrld)
    return closest_bomb(), closest_monster((coords[0], coords[1]), wrld), monster_direction(coords, wrld)

# ==================== FEATURES ==================== #
#   - Distance to closest bomb
#   - Distance to closest monster
#   - 1 / (Distance to exit)^2


# Returns an integer representing the Manhattan distance to the closest bomb.
def closest_bomb():
   return 1  # Will implement this later, in part 2 most likely. For now we don't care about bombs.


# Returns an integer representing the A* distance to the closest monster.
def closest_monster(coords, wrld):
    x = coords[0]
    y = coords[1]
    monsters = monster_tiles(wrld)
    p = float('inf')
    for m in monsters:
        distance = len(aStar((x, y), wrld, m))
        if distance < p:
            p = distance
    return p


# Returns 1/(A* distance to exit)^2.
def distance_to_exit(coords, wrld):
    dist = (len(aStar(coords, wrld, wrld.exitcell)) ** 2)
    if dist < 1:
        return 0
    return 1 / dist

def euclidean_distance(coords, wrld):
    return math.sqrt(((coords[0] - wrld.exitcell[0]) * coords[1] - wrld.exitcell[1]) ** 2)


# Returns a list of tiles which are occupied by at least 1 monster.
def monster_tiles(wrld):
    tiles = []
    for x in range(0, wrld.width()):
        for y in range(0, wrld.height()):
            if wrld.monsters_at(x, y):
                tiles.append((x, y))
    return tiles

def monster_direction(coords, wrld):
    x = coords[0]
    y = coords[1]
    monsters = monster_tiles(wrld)
    mcoords = 0, 0
    xval = 0
    yval = 0
    p = float('inf')
    for m in monsters:
        distance = len(aStar((x, y), wrld, m))
        if distance < p:
            p = distance
            mcoords = (m[0], m[1])
            xval = m[0] - x
            yval = m[1] - y

    return np.sign(xval), np.sign(yval)



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

def reward(c, wrld):
    dist = len(aStar((c.x, c.y), wrld, wrld.exitcell))
    if dist == 0:
        return 1
    return -dist

def aStar(start, wrld, goal):
    x = start[0]
    y = start[1]
    # print("SELFX: " + str(self.x))
    # print("SELFY: " + str(self.y))
    frontier = []
    frontier.append(((x, y), 0))
    came_from = {}
    cost_so_far = {}
    came_from[(x, y)] = None
    cost_so_far[(x, y)] = 0

    monsters = []


    while not len(frontier) == 0:
        frontier.sort(key=lambda tup: tup[1])  # check that
        current = frontier.pop(0)
        if (current[0][0], current[0][1]) == goal:
            break
        for next in get_adjacent(current[0], wrld):
            if wrld.wall_at(next[0], next[1]):
                cost_so_far[(next[0], next[1])] = 999
                new_cost = 1000
            else:
                new_cost = cost_to(current[0], next) + cost_so_far[current[0]]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                frontier.append((next, new_cost + manhattan_distance(next[0], next[1], goal[0], goal[1])))
                came_from[next] = current[0]

    cursor = goal
    path = []
    while not cursor == (x, y):
        path.append(cursor)
        try:
            cursor = came_from[cursor]
        except KeyError:
            return 0, 0
    #if draw:
        #print("PATH: ")
        #print(path)

    return path