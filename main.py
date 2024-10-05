import pygame
from pygame.math import Vector2 as Vec2
import sys
import cProfile
import random
from dataclasses import dataclass
from math import e, cos, sin, log10, floor, exp, pi
from enum import Enum
from pygame.time import get_ticks as ticks
import tomllib as toml
import re
from pprint import pprint
import numpy as np


# functions
def save_heatmap():
    global heatmap
    heatmap_surf = pygame.Surface((WIDTH, HEIGHT))
    heatmap /= heatmap.max()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            color = lerp_heatmap(heatmap[y, x])
            heatmap_surf.set_at((x, y), color)
    pygame.image.save(heatmap_surf, "heatmap.png")


def lerp_heatmap(i):
    if i < 0.5:
        r = 0
        g = i / 0.5 * 255
        b = 255 - (i / 0.5 * 255)
    else:
        r = (i - 0.5) / 0.5 * 255
        g = 255 - ((i - 0.5) / 0.5 * 255)
        b = 0
    return (r, g, b)


def parse_wait(wait):
    pattern = r"(\w+)\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)"
    match = re.match(pattern, str(wait))
    try:
        distribution = match.group(1).upper()
    except AttributeError:
        func = lambda: wait
    else:
        arg1 = float(match.group(2)) * 1000
        arg2 = float(match.group(3)) * 1000
        if distribution == "UNIFORM":
            func = lambda: random.uniform(arg1, arg2)
        elif distribution == "GAUSS":
            func = lambda: random.gauss(arg1, arg2)
    return func


def traverse_and_update(node, visited=None):
    if visited is None:
        visited = set()

    if node is None or node in visited:
        return

    node_obj = pool[node]
    
    node_obj.update()
    visited.add(node)
    traverse_and_update(node_obj.child, visited)


def dist_point_to_line_segment(p1, p2, pos):
    line_vec = p2 - p1
    pnt_vec = pos - Vec2(p1)
    line_len = line_vec.length()
    line_unitvec = line_vec.normalize()
    pnt_vec_scaled = pnt_vec / line_len
    t = line_unitvec.dot(pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = (nearest - pnt_vec).length()
    nearest = nearest + p1
    #
    normal = (pos - nearest)
    return (normal, dist)

def clamp(value, mn, mx):
    return max(mn, min(value, mx))


def sigfig(x: float, precision: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """
    x = float(x)
    precision = int(precision)
    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))
  

# classes
class Global:
    def __init__(self, name, target_fps):
        self.name = name
        self.target_fps = target_fps


class EditorModes(Enum):
    ERASE = -1
    POLYGON = 0
    REVOLVER = 1


class Editor:
    def __init__(self):
        self.points = []
        self.mode = EditorModes.POLYGON
    
    def process_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.points.append(pygame.mouse.get_pos())
            elif event.button == 3:
                if len(self.points) >= 2:
                    polygon = Polygon(*self.points)
                    all_obstacles.append(polygon)
                    self.points.clear()
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LCTRL:
                ...
                Pedestrian.move = not Pedestrian.move
                # if Pedestrian.move:
                #     Pedestrian.dest = pygame.mouse.get_pos()
                # else:
                #     Pedestrian.dest = None
    
    def update(self):
        m = Vec2(pygame.mouse.get_pos())
        mod = pygame.key.get_mods()
        if self.mode == EditorModes.POLYGON:
            if self.points:
                pygame.draw.lines(WIN, BLACK, False, self.points + [pygame.mouse.get_pos()], 5)


editor = Editor()


@dataclass
class Walk:
    mu: float = 1.40
    sigma: float = 0.36
    min: float = 0.5
    max: float = 2.2


class Spawner:
    def __init__(self, name, line, wait, limit=float("inf"), child=None):
        self.name = name
        self.line = line
        self.w = int(line[1][0] - line[0][0])
        self.h = int(line[1][1] - line[0][1])
        self.normal = Vec2(self.w, self.h).normalize().rotate(-90)
        self.get_wait = parse_wait(wait)
        self.wait = self.get_wait()
        self.last_time = ticks()
        self.child = child
        self.limit = limit
        self.spawned = 0
    
    def draw(self):
        for p in self.line:
            pygame.draw.circle(WIN, pygame.Color("#1167b1"), p, 5)
        pygame.draw.lines(WIN, pygame.Color("#2a9df4"), False, self.line)
    
    def update(self):
        if ticks() - self.last_time >= self.wait and self.spawned < self.limit:
            x = self.line[0][0] + random.randint(0, self.w)
            y = self.line[0][1] + random.randint(0, self.h)
            ped = Pedestrian(x, y)

            pool[self.child].new_ped(ped)
            all_pedestrians.append(ped)
            self.spawned += 1

            self.last_time = ticks()
            self.wait = self.get_wait()
        self.draw()


class Pedestrian:
    N = 20
    move = True
    dest = None
    def __init__(self, x, y, color=None):
        # init
        self.radius = 5
        self.pos = Vec2(x, y)
        self.gate = random.randrange(Gate.N)
        self.dest = Vec2(50, 50)
        self.color = color if color is not None else pygame.Color("#FDFBD4")
        self.color = color if color is not None else pygame.Color("#0F4C5C")
        self.waiting_color = pygame.Color("#990000")

        # driving term
        self.v0 = 2 * clamp(random.gauss(walk.mu, walk.sigma), walk.min, walk.max)
        self.vel = Vec2(0, 0)
        self.acc = Vec2(0, 0)
        self.dacc = Vec2(0, 0)
        self.oacc = Vec2(0, 0)
        self.iacc = Vec2(0, 0)
        self.t = 10
        # obstacle term
        self.A = 3
        self.r_o = 4
        # interactive term (with other pedestrians)
        self.B = 1
        self.r_i = 4
    
    def start_waiting(self):
        if not self.waiting:
            self.last_wait = ticks()
            self.waiting = True
    
    def calculate_drive_force(self):
        dest = Pedestrian.dest if Pedestrian.dest is not None else self.dest
        self.e = (dest - self.pos) / (dest - self.pos).length()
        desired_vel = self.v0 * self.e
        delta_vel = desired_vel - self.vel
        return 1 / self.t * delta_vel
    
    def calculate_obstacle_force(self):
        total_f = Vec2(0, 0)
        for ob in all_obstacles:
            for n, d in ob.get_distances(self):
                f = self.A * exp(-d / self.r_o) * n
                total_f += f
        return total_f
    
    def calculate_interactive_force(self):
        total_f = Vec2(0, 0)
        for ped in all_pedestrians:
            if ped is not self:
                d = (ped.pos - self.pos).length()
                n = (self.pos - ped.pos)
                f = self.B * exp(-d / self.r_i) * n
                total_f += f
        return total_f
    
    def update(self, dt):
        # update
        if Pedestrian.move:
            self.dacc = self.calculate_drive_force()
            self.oacc = self.calculate_obstacle_force()
            self.iacc = self.calculate_interactive_force()
            # newton
            self.acc = self.dacc + self.oacc + self.iacc
            self.vel += self.acc
            self.pos += self.vel * dt
        # draw
        pedestrians_to_draw.append(self)
        # update heatmap
        heatmap[int(self.pos.y), int(self.pos.x)] += 1
    
    def draw(self):
        color = self.color
        if self.waiting:
            color = self.waiting_color
        pygame.draw.aacircle(WIN, color, self.pos, self.radius)
        pygame.draw.aacircle(WIN, BLACK, self.pos, 5, 1)
        #
        m = 3
        pygame.draw.line(WIN, (0, 255, 0), self.pos, self.pos + self.vel * m, 2)
        pygame.draw.line(WIN, (255, 140, 0), self.pos, self.pos + self.acc * m * 7, 2)
    

class Gate:
    N = 5
    def __init__(self, x, y):
        self.image = pygame.Surface((20, 60))
        self.image.fill(DARK_GRAY)
        self.rect = self.image.get_frect(topleft=(x, y))
    
    @property
    def xy(self):
        return Vec2(self.rect.midbottom)
    
    def update(self):
        WIN.blit(self.image, self.rect)


class AbstractObstacle:
    def draw(self):
        WIN.blit(self.image, self.rect)


class Obstacle(AbstractObstacle):
    def __init__(self, x, y):
        self.image = pygame.Surface((24, 24), pygame.SRCALPHA)
        # pygame.draw.rect(self.image, BLACK, (0, 0, *self.image.size), 2)
        pygame.draw.circle(self.image, BLACK, [s / 2 for s in self.image.size], self.image.width / 2, 2)
        self.rect = self.image.get_frect(topleft=(x, y))
    
    def update(self):
        self.draw()
    
    @property
    def xy(self):
        return Vec2(self.rect.center)
    
    def get_distances(self, other):
        normal = (other.pos - self.xy)
        return [(normal, normal.length())]
    

class Polygon(AbstractObstacle):
    def __init__(self, name, points):
        self.name = name
        self.points = [Vec2(p) for p in points]
    
    def update(self):
        self.draw()
    
    def draw(self):
        pygame.draw.lines(WIN, BLACK, False, self.points, 5)
    
    def get_distances(self, other):
        for i in range(len(self.points)):
            p1 = self.points[i]
            try:
                p2 = self.points[i + 1]
            except IndexError:
                continue
            yield dist_point_to_line_segment(p1, p2, other.pos)


class Revolver(AbstractObstacle):
    def __init__(self, p1, n, l, av):
        self.p1 = Vec2(p1)
        self.n = n
        self.l = l
        self.av = av
        self.angle = 0
        self.p2s = [Vec2(0, 0) for _ in range(self.n)]
    
    def update(self):
        self.angle += self.av
        for i in range(self.n):
            a = self.angle + i * (2 * pi) / self.n
            arm = self.l * Vec2(cos(a), sin(a))
            p2 = self.p1 + arm
            self.p2s[i] = p2
            pygame.draw.line(WIN, BLACK, self.p1, p2, 5)
            
    def get_distances(self, other):
        for p2 in self.p2s:
            yield dist_point_to_line_segment(self.p1, p2, other.pos)


class Queue:
    def __init__(self, points):
        self.points = points
        self.available_point = 0
    
    def get_point(self):
        self.available_point += 1
        return self.points[self.available_point - 1]
    
    def update(self):
        for point in self.points:
            pygame.draw.circle(WIN, DARK_GRAY, point, 5)


class Area:
    def __init__(self, name, area, dimensions, wait=None, child=None):
        self.name = name
        self.area = area
        self.x, self.y, self.w, self.h = self.area
        self.dimensions = dimensions
        self.num_x, self.num_y = self.dimensions
        self.attractors = []
        for y in range(self.num_y):
            for x in range(self.num_x):
                self.attractors.append((self.x + (x + 0.5) / self.num_x * self.w, self.y + (y + 0.5) / self.num_y * self.h))
        self.child = child
        self.last_time = ticks()
        self.pedestrians = []
        self.get_wait = parse_wait(wait)
    
    def new_ped(self, ped):
        ped.waiting = False
        ped.dest = random.choice(self.attractors)
        ped.wait = self.get_wait()
        self.pedestrians.append(ped)

    def draw(self):
        pygame.draw.rect(WIN, (170, 170, 170), self.area)
        pygame.draw.rect(WIN, (100, 100, 100), self.area, 3)
        for pos in self.attractors:
            pygame.draw.circle(WIN, pygame.Color("#B2D3C2"), pos, 5)

    def update(self):
        self.draw()
        for ped in self.pedestrians.copy():
            ped.update(dt=1)
            # does pedestrian need to start waiting?
            if (ped.dest - ped.pos).length() <= 0.3:
                ped.start_waiting()
            # does pedestrian need to go to next place?
            if self.child is not None and ped.waiting and ticks() - ped.last_wait >= ped.wait:
                self.pedestrians.remove(ped)
                pool[self.child].new_ped(ped)


# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (200, 200, 200)
SMALL = 0.0001
# constants
pygame.init()
WIDTH = 1000
HEIGHT = 800
pygame.display.set_caption("Social Force Model")
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Courier", 20)
heatmap = np.zeros((HEIGHT, WIDTH))

# sim initialization
walk = Walk()
all_obstacles = []
all_pedestrians = []
pedestrians_to_draw = []

# load the model file
pool = {}
entry = None
static_objects = []
with open("model.absml", "rb") as f:
    data = toml.load(f)
    for k, v in data.items():
        v |= {"name": k}
        if k == "global":
            g = Global(**v)
        elif k.startswith("spawner"):
            spawner = Spawner(**v)
            pool[k] = spawner
        elif k.startswith("area"):
            area = Area(**v)
            pool[k] = area
        elif k.startswith("polygon"):
            polygon = Polygon(**v)
            pool[k] = polygon
            all_obstacles.append(polygon)
        if k.endswith("!"):
            entry = k


def main():
    # mainloop
    running = __name__ == "__main__"
    while running:
        dt = clock.tick(g.target_fps) / 1000 / (1 / 120)
    
        for event in pygame.event.get():
            editor.process_event(event)

            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    save_heatmap()
                    running = False
    
        # clearing window
        WIN.fill(LIGHT_GRAY)

        # updating the simulation

    
        pedestrians_to_draw.clear()
        traverse_and_update(entry)
        for ped in pedestrians_to_draw:
            ped.draw()
        
        for ob in all_obstacles:
            ob.update()
     
        # displaying fps (veri important)
        surf = font.render(str(int(clock.get_fps())), True, BLACK)
        WIN.blit(surf, (10, 10))

        editor.update()
    
        # flip the display
        pygame.display.flip()

    pygame.quit()
    sys.exit()

main()