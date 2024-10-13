import pygame
from pygame.math import Vector2 as Vec2
import sys
import cProfile
import random
from dataclasses import dataclass
from math import e, cos, sin, log10, floor, exp, pi, degrees, radians
from enum import Enum
from pygame.time import get_ticks as ticks
import tomllib as toml
import re
from pprint import pprint
import numpy as np
from pathlib import Path
from contextlib import suppress
from itertools import product


# functions
def save(toml_path):
    with open(toml_path, "w") as toml_file:
        toml_file.write(g.get_toml())
        for obj in pool.values():
            toml_file.write(obj.get_toml())

    pygame.quit()
    sys.exit()


def load_model(path):
    global g, all_obstacles, all_spawners, static_objects, pool

    all_obstacles = []
    static_objects = []
    all_spawners = []
    file_vars = {}

    with open(path, "rb") as f:
        data = toml.load(f)
        for k, v in data.items():
            if k == "vars":
                file_vars |= v
            v |= {"name": k}

            if k == "global":
                g = Global(**v, edit=False)
            elif k.startswith("spawner"):
                spawner = Spawner(**v)
                pool[k] = spawner
                all_spawners.append(k)
            elif k.startswith("area"):
                area = Area(**v)
                pool[k] = area
            elif k.startswith("polygon"):
                polygon = Polygon(**v)
                pool[k] = polygon
                all_obstacles.append(polygon)


def grid_to_pos(x, y):
    return [x * g.grid, g.height - (y * g.grid)]


def roundn(x, base):
    return base * round(x / base)


def save_heatmap():
    global heatmap
    heatmap_surf = pygame.Surface((g.width, g.height))
    heatmap /= heatmap.max()
    for y in range(g.height):
        for x in range(g.width):
            color = lerp_heatmap(heatmap[y, x])
            heatmap_surf.set_at((x, y), color)
    # pygame.image.save(heatmap_surf, Path("res", "heatmap.png"))


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
        arg1 = float(match.group(2))
        arg2 = float(match.group(3))
        if distribution == "UNIFORM":
            func = lambda: random.uniform(arg1, arg2)
        elif distribution == "GAUSS":
            func = lambda: random.gauss(arg1, arg2)
    return func


def traverse_and_update(node, visited_global, visited=None):
    if visited is None:
        visited = set()
    if node is None or node in visited or node in visited_global:
        return

    node_obj = pool[node]
    node_obj.update()

    visited.add(node)
    visited_global.add(node)

    for child in node_obj.children:
        traverse_and_update(child, visited_global, visited)


def traverse_and_save_toml(node, visited_global, toml_file, visited=None):
    if visited is None:
        visited = set()
    if node is None or node in visited or node in visited_global:
        return

    node_obj = pool[node]

    toml_file.write(node_obj.get_toml())

    visited.add(node)
    visited_global.add(node)
    for child in node_obj.children:
        traverse_and_save_toml(child, visited_global, toml_file, visited)


def dist_point_to_line_segment(p1, p2, pos):
    line_vec = p2 - p1
    pnt_vec = pos - Vec2(p1)
    line_len = line_vec.length()
    try:
        line_unitvec = line_vec.normalize()
    except ValueError:
        line_unitvec = Vec2(1, 1)
    pnt_vec_scaled = pnt_vec / (line_len + 0.01)
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
    def __init__(self, name, target_fps, width, height, draw=True, edit=None):
        self.name = name
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.grid = 30
        self.draw = draw
        self.edit = edit
    
    def get_toml(self):
        ret = f"[global]\n"
        ret += f"width = {self.width}\n"
        ret += f"height = {self.height}\n"
        ret += f"target_fps = {self.target_fps}\n"
        ret += f"draw = {str(self.draw).lower()}\n"
        ret += "\n"
        return ret
    
    def main(self):
        # mainloop
        running = __name__ == "__main__"
        while running:
            dt = clock.tick(self.target_fps) / 1000 / (1 / 120)
        
            for event in pygame.event.get():
                editor.process_event(event)

                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # save_heatmap()
                        running = False
                    
                    elif event.key == pygame.K_e:
                        g.draw = not g.draw
                
                elif event.type == pygame.MOUSEWHEEL:
                    editor.vec_angle += event.y
        
            # clearing window
            if self.edit:
                with suppress(Exception):
                    load_model("model.toml")
            WIN.blit(grid_surf, (0, 0))

            # updating the simulation
        
            pedestrians_to_draw.clear()

            visited_global = set()

            if g.draw:
                for name, obj in pool.items():
                    obj.draw()
            else:
                for spawner in all_spawners:
                    traverse_and_update(spawner, visited_global)

            for ped in pedestrians_to_draw:
                ped.draw()
            
            for ob in all_obstacles:
                ob.update()
            
            for (x, y), vector in vector_field.items():
                vector.update(x, y)
        
            # displaying fps (veri important)
            text = f"{int(clock.get_fps())}\nPedestrians: {len(pedestrians_to_draw)}"
            surf = font.render(text, True, BLACK)
            WIN.blit(surf, (10, 10))
            if g.draw:
                surf = font.render("[DRAW MODE]", True, BLACK)
                rect = surf.get_rect(midtop=(g.width / 2, 60))
                WIN.blit(surf, rect)

            editor.update()
        
            # flip the display
            pygame.display.flip()

        save("save.toml")


class EditorModes(Enum):
    ERASE = -1
    POLYGON = 0
    RECT = 1
    AREA = 2
    SPAWNER = 3
    VECTOR = 4


class Editor:
    def __init__(self):
        self.points = []
        self.area = []
        self.mode = EditorModes.POLYGON
        self.vec_angle = 0
    
    def process_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.mode == EditorModes.POLYGON:
                    self.points.append(self.placing_pos)

                elif self.mode == EditorModes.SPAWNER:
                    if not self.points:
                        self.points.append(self.placing_pos)
                    else:
                        self.points.append(self.placing_pos)
                        name = Spawner.get_name()
                        spawner = Spawner(name, self.points, 100)
                        pool[name] = spawner
                        self.points.clear()

                elif self.mode == EditorModes.AREA:
                    if not self.area:
                        self.area = self.placing_pos
                    else:
                        x, y = self.placing_pos
                        w, h = x - self.area[0], y - self.area[1]
                        name = Area.get_name()
                        area = Area(name, [*self.area, w, h], [5, 5], kill=True)
                        pool[name] = area
                        self.area = []
                
                elif self.mode == EditorModes.RECT:
                    if not self.area:
                        self.area = self.placing_pos
                    else:
                        points = [
                            (self.area[0], self.area[1]),
                            (self.placing_pos[0], self.area[1]),
                            (self.placing_pos[0], self.placing_pos[1]),
                            (self.area[0], self.placing_pos[1]),
                        ]
                        name = Polygon.get_name()
                        poly = Polygon(name, points, connect=True)
                        pool[name] = poly
                        self.area = []
                
                elif self.mode == EditorModes.VECTOR:
                    v = FieldVector(radians(editor.vec_angle))
                    indexes = tuple(p / g.grid for p in self.placing_pos)
                    vector_field[indexes] = v

            elif event.button == 3:
                if self.mode == EditorModes.POLYGON:
                    if self.points:
                        del self.points[-1]

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LCTRL:
                Pedestrian.move = not Pedestrian.move
            
            elif event.key == pygame.K_RETURN:
                if len(self.points) >= 2:
                    name = Polygon.get_name()
                    polygon = Polygon(name, points=self.points.copy())
                    pool[name] = polygon
                    self.points.clear()
            
            else:
                self.mode = {
                    49: EditorModes.POLYGON,
                    50: EditorModes.RECT,
                    51: EditorModes.AREA,
                    52: EditorModes.SPAWNER,
                    53: EditorModes.VECTOR
                }.get(event.key, self.mode)
    
    def update(self):
        self.placing_pos = [roundn(p, 30) for p in pygame.mouse.get_pos()]
        pygame.draw.circle(WIN, BLACK, self.placing_pos, 4)
        if self.mode == EditorModes.POLYGON:
            if self.points:
                pygame.draw.lines(WIN, (100, 100, 100, 255), False, self.points + [self.placing_pos], 5)
        
        elif self.mode == EditorModes.SPAWNER:
            if self.points:
                pygame.draw.lines(WIN, pygame.Color("dark blue"), False, self.points + [self.placing_pos], 3)

        elif self.mode in (EditorModes.RECT, EditorModes.AREA):
            if self.mode == EditorModes.RECT:
                color = BLACK
            else:
                color = pygame.Color("sky blue")
            s = 30
            if self.area:
                x, y = self.placing_pos
                w, h = x - self.area[0], y - self.area[1]
                if self.mode == EditorModes.AREA:
                    pygame.draw.rect(WIN, pygame.Color("#1167b1"), (*self.area, w, h))
                pygame.draw.rect(WIN, color, (*self.area, w, h), 5)
        
        elif self.mode == EditorModes.VECTOR:
            image = pygame.transform.rotozoom(vector_image, self.vec_angle, 1)
            rect = image.get_rect(center=(g.width - 20, 20))
            WIN.blit(image, rect)
        
        mode_surf = font.render(str(self.mode), True, BLACK)
        mode_rect = mode_surf.get_rect(midtop=(g.width / 2, 30))
        WIN.blit(mode_surf, mode_rect)


editor = Editor()


@dataclass
class Walk:
    mu: float = 1.40
    sigma: float = 0.36
    min: float = 0.5
    max: float = 2.2


class Node:
    @classmethod
    def get_name(cls):
        num = 0
        name = f"{cls.__name__.lower()}{num}"
        while True:
            for other_name in pool.keys():
                name = f"{cls.__name__.lower()}{num}"
                if other_name == name:
                    num += 1
                    break
            else:
                break
        return name

    def get_child(self):
        try:
            return random.choices(self.children, self.chances, k=1)[0]
        except ValueError:
            raise RuntimeError("Target node has multiple children but no `chances` distribution")
    
    def get_toml(self):
        ret = f"[{self.name}]\n"
        for attr in self.toml_attrs:
            attr_obj = getattr(self, attr)
            if attr_obj is None:
                continue
            if isinstance(attr_obj, str):
                attr_obj = f'"{attr_obj}"'
            if isinstance(attr_obj, bool):
                attr_obj = str(attr_obj).lower()
            if attr == "line":
                attr_obj = getattr(self, "grid_line", self.line)
            if isinstance(attr_obj, list) and attr_obj and isinstance(attr_obj[0], Vec2):
                attr_obj = [[vec.x, vec.y] for vec in attr_obj]
            ret += f"{attr} = {attr_obj}\n"
        ret += "\n"
        return ret


class Spawner(Node):
    def __init__(self, name, line, wait, limit=10 ** 5, color=None, children=None, chances=None):
        self.toml_attrs = ("line", "wait", "limit", "color", "children", "chances")
        self.name = name
        self.line = [Vec2(*p) for p in line]
        self.w = int(self.line[1][0] - self.line[0][0])
        self.h = int(self.line[1][1] - self.line[0][1])
        self.normal = Vec2(self.w, self.h).normalize().rotate(-90)
        self.get_wait = parse_wait(wait)
        self.wait = self.get_wait()
        self.last_time = ticks()
        self.children = children if children is not None else []
        self.chances = chances if chances is not None else [1]
        self.limit = limit
        self.spawned = 0
        self.color = color
    
    def draw(self):
        for p in self.line:
            pygame.draw.circle(WIN, pygame.Color("#1167b1"), p, 5)
        pygame.draw.lines(WIN, pygame.Color("#2a9df4"), False, self.line)
    
    def update(self):
        if not g.edit:
            if self.children and ticks() - self.last_time >= self.wait and self.spawned < self.limit:
                x = self.line[0][0] + random.randint(0, self.w)
                y = self.line[0][1] + random.randint(0, self.h)
                ped = Pedestrian(x, y, color=self.color)

                pool[self.get_child()].new_ped(ped)
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
        self.v0 = 0.9 * clamp(random.gauss(walk.mu, walk.sigma), walk.min, walk.max)
        self.pving = False
        self.vel = Vec2(0, 0)
        self.acc = Vec2(0, 0)
        self.dacc = Vec2(0, 0)
        self.oacc = Vec2(0, 0)
        self.iacc = Vec2(0, 0)
        self.t = 10
        # obstacle term
        self.A = 3
        self.r_o = 2
        # interactive term (with other pedestrians)
        self.B = 1
        self.r_i = 4
    
    def start_waiting(self):
        if not self.waiting:
            self.last_wait = ticks()
            self.waiting = True
    
    def start_pv(self):
        self.start_waiting()
        self.pving = True
    
    def calculate_drive_force(self):
        grid_x = int(self.pos.x / g.grid)
        grid_y = int(self.pos.y / g.grid)
        grid_pos = (grid_x, grid_y)
        if grid_pos in vector_field:
            angle = vector_field[grid_pos].angle
            self.e = Vec2(cos(angle), -sin(angle))
            # self.color = pygame.Color("orange")
        else:
            dest = Pedestrian.dest if Pedestrian.dest is not None else self.dest
            self.e = (dest - self.pos) / (dest - self.pos).length()
            # self.color = pygame.Color("lavender")
        desired_vel = (0 if self.pving else self.v0) * self.e
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
                if (ped.pos - self.pos).length() >= 30:
                    continue
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
        # heatmap[int(self.pos.y), int(self.pos.x)] += 1
    
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


class AbstractObstacle(Node):
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
    def __init__(self, name, points, connect=False, queue=False):
        self.toml_attrs = ("points", "connect", "queue")
        self.name = name
        self.points = [Vec2(*p) for p in points]
        self.connect = connect
        if self.connect:
            self.points.append(self.points[0])
        self.queue = queue
    
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


class FieldVector(Node):
    def __init__(self, angle):
        self.angle = angle
        self.image = pygame.transform.rotozoom(vector_image, degrees(angle), 1)
        self.rect = vector_image.get_rect()
    
    def update(self, xindex, yindex):
        self.rect.center = ((xindex + 0.5) * g.grid, (yindex + 0.5) * g.grid)
        WIN.blit(self.image, self.rect)


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


class Area(Node):
    def __init__(self, name, area, dimensions, wait_mode="pv", wait=None, kill=False, children=None, chances=None):
        self.toml_attrs = ("area", "dimensions", "wait_mode", "wait", "kill", "children", "chances")
        self.name = name
        self.area = area
        self.rect = pygame.Rect(area)

        self.dimensions = dimensions
        self.num_x, self.num_y = self.dimensions
        self.attractors = []

        self.children = children if children is not None else []
        self.chances = chances if chances is not None else [1]
        self.last_time = ticks()
        self.pedestrians = []
        self.kill = kill
        self.get_wait = parse_wait(wait)
        self.wait_mode = wait_mode
        self.wait = wait
    
    def new_ped(self, ped):
        ped.waiting = False
        if self.wait_mode == "att":
            if self.attractors:
                ped.dest = random.choice(self.attractors)
            else:
                ped.dest = self.rect.center
        elif self.wait_mode == "pv":
            ped.dest = self.rect.center
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
            if self.wait_mode == "att":
                if (ped.dest - ped.pos).length() <= 3:
                    if self.kill:
                        self.pedestrians.remove(ped)
                        all_pedestrians.remove(ped)
                    else:
                        ped.start_waiting()
            elif self.wait_mode == "pv":
                if self.rect.collidepoint(ped.pos):
                    ped.start_pv()
            # does pedestrian need to go to next place?
            if self.children and ped.waiting:
                if ped.wait is None:
                    raise RuntimeError(f"{self.name} object hasn't been given a `wait` attribute, but has children")

                if ticks() - ped.last_wait >= ped.wait:
                    self.pedestrians.remove(ped)
                    pool[self.get_child()].new_ped(ped)

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (200, 200, 200)
SMALL = 0.0001

walk = Walk()
all_pedestrians = []
pedestrians_to_draw = []
all_obstacles = []
all_spawners = []
static_objects = []
g = None
pool = {}
g = Global("global", 120, 810, 810)
load_model("save.toml")
vector_image = pygame.transform.flip(pygame.transform.scale(pygame.image.load(Path("../res", "arrow.png")), (g.grid * 0.6, g.grid * 0.6)), True, False)

grid_surf = pygame.Surface((g.width, g.height))
grid_surf.fill(LIGHT_GRAY)
for y in range(g.height // g.grid):
    for x in range(g.width // g.grid):
        pygame.draw.line(grid_surf, (140, 140, 140), (x * g.grid, 0), (x * g.grid, g.height))
    pygame.draw.line(grid_surf, (140, 140, 140), (0, y * g.grid), (g.width, y * g.grid))

# constants
pygame.init()
pygame.display.set_caption("Social Force Model")
WIN = pygame.display.set_mode((g.width, g.height), pygame.RESIZABLE)
clock = pygame.time.Clock()
font = pygame.font.SysFont("Courier", 20)
heatmap = np.zeros((g.height, g.width))

vector_field = {}
# for y in range(g.height // g.grid):
#     for x in range(g.width // g.grid):
#         if 5 <= x <= 15 and 10 <= y <= 14:
#             vf = ValueError
#             vector_field[(x, y)] = FieldVector(random.uniform(0, 2 * pi))
            
g.main()