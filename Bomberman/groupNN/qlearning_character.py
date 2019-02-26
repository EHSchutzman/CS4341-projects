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

    def __init__(self, qtable, wb, wm, wg, ww, *args, **kwargs):
        super(QCharacter, self).__init__(*args, **kwargs)
        # Whether this character wants to place a bomb
        self.maybe_place_bomb = False
        # Debugging elements
        self.tiles = {}
        self.qtable = qtable
        self.wb = wb  # weight of bomb feature
        self.wm = wm  # weight of monster distance feature
        self.wg = wg  # weight of goal distance
        self.ww = ww  # weight of distance to closest wall
        self.last_action = None

        # return closest_bomb(), closest_monster((coords[0], coords[1]), wrld), monster_direction(coords, wrld), dist

    def do(self, wrld):
        if not self.threatened(wrld):
            path = aStar((self.x, self.y), wrld, wrld.exitcell)  # (7, 18) usualy
            print("Path " + str(path))
            move = path[len(path) - 1]
            print("move " + str(move))
            self.move(move[0] - self.x, move[1] - self.y)
            self.last_action = move[0] - self.x, move[1] - self.y
            pass
        else:
            # IF threatened we want to
            # 1st move away from monster
            # THEN move toward goal.
            # QLEARNING FOR THIS???
            state = calculate_state((self.x, self.y), wrld)
            print("NExt best score::: ")
            print(self.getNextBestScore(state, wrld))

            actions = self.valid_moves(wrld)
            for a in actions:
                 self.approximateQ(state, a, wrld)

            move = self.select_best_move(state, actions, wrld)

            print("MOVE:")
            print(move)

            print("WEIGHTS:")
            print(self.wb)
            print(self.wm)
            print(self.wg)

            self.last_action = move
            self.move(move[0], move[1])
            pass

    def approximateQ(self, state, action, wrld):
        alpha = 0.3
        gamma = 0.2
        keys = self.qtable.keys()
        ra = (action[0] - self.x, action[1] - self.y)  # Relative action
        if (state, ra) not in keys:
            self.qtable[(state, ra)] = 0

        # Update weights of each feature
        # TODO FIX self.getNextBestScore
        delta = (gamma * self.getNextBestScore(state, wrld)) - self.qtable[state, ra]
        # First feature: distance to bomb
        self.wb = self.wb + alpha * delta * closest_bomb()
        # Second feature: distance to closest monster
        self.wm = self.wm + alpha * delta * closest_monster((self.x, self.y), wrld)
        # Third feature: direction of closest monster, DOESN'T ACTUALLY WORK BC NOT A SCALAR

        # Fourth feature: distance to exit
        self.wg = self.wg + alpha * delta * distance_to_exit((self.x, self.y), wrld)

        # Fourth feature: distance to exit
        self.ww = self.ww + alpha * delta * closest_wall((self.x, self.y), wrld)

        self.qtable[(state, ra)] = self.qtable[(state, ra)] + delta*alpha

    # Pretty sure there is a problem here, with simulating next worlds.
    def getNextBestScore(self, state, wrld):
        actions = get_adjacent((self.x, self.y), wrld)
        keys = self.qtable.keys()

        # Find best known score in qtable
        bestscore = float('-inf')
        potential = []

        for a in actions:
            # Check that there isn't a wall at this move
            if not wrld.wall_at(a[0], a[1]):
                # Check the reward we end up at after making this move
                # Initialize state value to 0 if we haven't seen it before
                ra = a[0] - self.x, a[1] - self.y  # RELATIVE action based on our position
                if (state, ra) not in keys:
                    self.qtable[(state, ra)] = 0
                # This is the score of the move from this state
                score = self.qtable[(state, ra)]
                if score > bestscore:
                    bestscore = score

        return bestscore


    def select_best_move(self, state, moves, wrld):
        candidates = []
        # Construct table keys from possible moves and current state.
        for m in moves:
            if not wrld.wall_at(m[0], m[1]):
                rm = (m[0] - self.x, m[1] - self.y)
                print(rm)  # Relative move
                candidates.append((state, rm))

        m = float('-inf')

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
        if closest_monster((self.x, self.y), wrld) <= 3:
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
    dist = distance_to_exit(coords, wrld)

    # TODO Add distance to wall??
    return closest_bomb(), monster, dist, closest_wall(coords, wrld)

# ==================== FEATURES ==================== #
#   - Distance to closest bomb
#   - Distance to closest monster
#   - Distance to goal
#   - Distance to closest wall

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
        return 1
    return 1 / dist

def closest_wall(coords, wrld):
    walls = get_walls(wrld)
    mindist = float("inf")
    for w in walls:
        dist = manhattan_distance(coords[0], coords[1], w[0], w[1])
        if dist < mindist:
            mindist = dist
    return mindist

# ========== END OF FEATURES ==========


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

def get_walls(wrld):
    walls = []

    for x in range(0, wrld.width()):
        for y in range(0, wrld.height()):
            if wrld.wall_at(x, y):
                walls.append((x, y))
    return walls

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

    while not len(frontier) == 0:
        frontier.sort(key=lambda tup: tup[1])  # check that
        current = frontier.pop(0)
        if (current[0][0], current[0][1]) == goal:
            break
        for next in get_adjacent(current[0], wrld):
            #print(next)
            if wrld.wall_at(next[0], next[1]):
                cost_so_far[(next[0], next[1])] = 999
                new_cost = 1000
            else:
                new_cost = cost_to(current[0], next) + cost_so_far[current[0]]
                #new_cost = 1 + cost_so_far[current[0]]
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
            return [(0, 0)]
    return path