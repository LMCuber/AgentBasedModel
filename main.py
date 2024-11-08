import pygame
from pygame.math import Vector2 as Vec2
import sys
import cProfile
import random
from dataclasses import dataclass
from math import e, cos, sin, log10, floor, exp, pi, degrees, radians
from math import log as ln
from enum import Enum
from pygame.time import get_ticks as ticks
import tomllib as toml
import re
from pprint import pprint
import numpy as np
from pathlib import Path
from contextlib import suppress
from itertools import product
import matplotlib.pyplot as plt
from time import perf_counter


# functions
def weibull(u, l, k):
    return l * (-ln(1 - u)) ** (1 / k)


def wait_from_weibull(l, k, unit_min, unit_max, scale):
    while not (unit_min <= (u := random.random()) <= unit_max):
        pass
    if u == 1:
        u -= 0.0000001
    t = weibull(u, l, k)
    value_min = weibull(0.00000001, l, k)
    value_max = weibull(0.99999999, l, k)
    ratio = t / value_max
    value = (1 - ratio) * scale
    return value


# plt.hist([wait_from_weibull(2.3, 6, 0.000001, 0.999, 200) for _ in range(1_000_000)], bins=100)
# plt.gca().invert_xaxis()
# plt.gca().set_xlim([200, 0])
# plt.show()
# raise


def save(toml_path):
    with open(toml_path, "w") as toml_file:
        toml_file.write(g.get_toml())
        for obj in pool.values():
            toml_file.write(obj.get_toml())
        toml_file.write("# VECTOR FIELDS\n\n")
        for i, (pos, vector) in enumerate(vector_field.items()):
            toml_file.write(f"[vector{i}]\n")
            toml_file.write(f"pos = {list(pos)}\n")
            toml_file.write(f"angle = {vector.angle}\n")
            toml_file.write("\n")

    pygame.quit()
    sys.exit()


def load_model(path):
    global g, all_obstacles, all_spawners, static_objects, pool, vector_field

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
            elif k.startswith("vector"):
                pos = tuple(v["pos"])
                del v["pos"]
                del v["name"]
                vector = FieldVector(**v)
                vector_field[pos] = vector
            elif k.startswith("revolver"):
                rev = Revolver(**v)
                pool[k] = rev
                all_obstacles.append(rev)


def grid_to_pos(x, y):
    return [x * g.grid, g.height - (y * g.grid)]


def roundn(x, base):
    return base * round(x / base)


def save_heatmap():
    global heatmap
    heatmap_surf = pygame.Surface((g.width // g.grid, g.height // g.grid))
    heatmap /= heatmap.max()
    for y in range(heatmap_surf.height):
        for x in range(heatmap_surf.width):
            color = lerp_heatmap(heatmap[y, x])
            heatmap_surf.set_at((x, y), color)
    pygame.image.save(heatmap_surf, Path("res", "heatmap.png"))


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
        self.last = ticks()
        self.running = False
        self.last_start = ticks()
        self.i = 120
    
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
        g.running = __name__ == "__main__"
        while g.running:
            dt = clock.tick(g.i) / 1000 / (1 / 120)
        
            for event in pygame.event.get():
                editor.process_event(event)

                if event.type == pygame.QUIT:
                    g.running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # save_heatmap()
                        g.running = False
                    
                    elif event.key == pygame.K_q:
                        print(ticks() - g.last)
                        g.running = False
                    
                elif event.type == pygame.MOUSEWHEEL:
                    editor.vec_angle += event.y * 4
                    g.i += event.y
                    g.i = max(0, g.i)
                
            # clearing window
            if self.edit:
                with suppress(Exception):
                    load_model(Path("src", "model.toml"))
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
            
            for (x, y), vector in vector_field.items():
                vector.update(x, y)
        
            for ped in pedestrians_to_draw:
                ped.draw()
            
            for ob in all_obstacles:
                if not g.draw:
                    ob.update()
                ob.draw()
   
            # displaying fps (veri important)
            text = f"{int(clock.get_fps())}\nPedestrians: {len(pedestrians_to_draw)}"
            surf = font.render(text, True, BLACK)
            WIN.blit(surf, (10, 10))
            if g.draw:
                surf = font.render("[DRAW MODE]", True, BLACK)
                rect = surf.get_rect(midtop=(g.width / 2, 60))
                WIN.blit(surf, rect)
            
            # display the time
            elapsed_ms = (ticks() - self.last_start)
            hours = floor(elapsed_ms / (3.6 * 10 ** 6) % 24)
            minutes = floor(elapsed_ms / 60_000 % 60)
            seconds = floor(elapsed_ms / 1000 % 60)
            millis = floor(elapsed_ms % 1000)
            time = f"{hours if len(str(hours)) > 1 else "0" + str(hours)}:{minutes if len(str(minutes)) > 1 else "0" + str(minutes)}:{seconds if len(str(seconds)) > 1 else "0" + str(seconds)}"
            surf = font.render(str(time), True, BLACK)
            WIN.blit(surf, (200, 10))

            editor.update()
        
            # flip the display
            pygame.display.flip()

        save(model_path)


class EditorModes(Enum):
    ERASE = -1
    POLYGON = 0
    RECT = 1
    AREA = 2
    SPAWNER = 3
    VECTOR = 4
    REVOLVER = 5


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
                    indexes = tuple(p // g.grid for p in self.placing_pos)
                    print(indexes)
                    vector_field[indexes] = v
                
                elif self.mode == EditorModes.REVOLVER:
                    name = Revolver.get_name()
                    r = Revolver(name, self.placing_pos, 4, 40, 0.25)
                    pool[name] = r

            elif event.button == 3:
                if self.mode == EditorModes.POLYGON:
                    if self.points:
                        del self.points[-1]
                
                elif self.mode == EditorModes.VECTOR:
                    indexes = tuple(p // g.grid for p in self.placing_pos)
                    # del vector_field[indexes]

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
                    53: EditorModes.VECTOR,
                    54: EditorModes.REVOLVER,
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


        self.flight_times = {30: []}
        for before in self.flight_times.keys():
            for _ in range(30):
                x = wait_from_weibull(2.3, 6, 0.000001, 0.999, 30)
                self.flight_times[before].append(x)
        print(self.flight_times)
    
    def draw(self):
        for p in self.line:
            pygame.draw.circle(WIN, pygame.Color("#1167b1"), p, 5)
        pygame.draw.lines(WIN, pygame.Color("#2a9df4"), False, self.line)
    
    def update(self):
        if not g.edit:
            if self.children and ticks() - self.last_time >= 400:
                if self.spawned < self.limit:
                    x = self.line[0][0] + random.randint(0, self.w)
                    y = self.line[0][1] + random.randint(0, self.h)
                    ped = Pedestrian(x, y, color=self.color)

                    pool[self.get_child()].new_ped(ped)
                    all_pedestrians.append(ped)
                    self.spawned += 1

                    self.last_time = ticks()
                    self.wait = self.get_wait()
                else:
                    pass

            for flight_time, times in self.flight_times.items():
                continue
                time_until_flight = flight_time - (ticks() - g.last_start) / 1000
                for i, time in enumerate(times):
                    if time_until_flight <= time:
                        x = self.line[0][0]
                        y = self.line[0][1]
                        ped = Pedestrian(x, y, color=self.color)

                        pool[self.get_child()].new_ped(ped)
                        all_pedestrians.append(ped)
                        del times[i]
                        self.spawned += 1

        self.draw()


class Pedestrian:
    N = 20
    move = True
    dest = None
    def __init__(self, x, y, color=None):
        # init
        self.pos = Vec2(x, y)
        self.gate = random.randrange(Gate.N)
        self.dest = Vec2(50, 50)
        # self.color = color if color is not None else pygame.Color("#FDFBD4")
        self.def_color = self.color = random.choice(palette)
        self.waiting_color = pygame.Color("#990000")
        # driving term
        self.v0 = clamp(random.gauss(walk.mu, walk.sigma), walk.min, walk.max)
        self.v0 = 1.8
        self.pving = False
        self.vel = Vec2(0, 0)
        self.acc = Vec2(0, 0)
        self.dacc = Vec2(0, 0)
        self.oacc = Vec2(0, 0)
        self.iacc = Vec2(0, 0)
        self.t = 10
        # general term
        self.r = 0.25 * g.grid
        # obstacle term
        self.A_ob = 3
        self.B_ob = 1
        self.follow_vectors = True
        # interactive term (with other pedestrians)
        self.A_ped = 0.08
        self.B_ped = 4
    
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
        if self.area.wait_mode == "queue":
            # ONLY WHEN IN A QUEUE!
            dest_rect = pygame.Rect((self.dest[0] - g.grid / 2, self.dest[1] - g.grid / 2, g.grid, g.grid))
            pygame.draw.rect(WIN, self.color, dest_rect)
            # check if it has to follow ground arrows (vector field)
            if self.follow_vectors and grid_pos in vector_field:
                vector = vector_field[grid_pos]
                angle = vector.angle
                self.e = Vec2(cos(angle), -sin(angle))
            else:
                self.e = (self.dest - self.pos) / (self.dest - self.pos).length()
            # check if colliding with dest
            if dest_rect.collidepoint(self.pos):
                # check if there are empty venues
                pass
        else:
            self.e = (self.dest - self.pos) / (self.dest - self.pos).length()
            self.color = self.def_color
       
        desired_vel = (0 if self.pving else self.v0) * self.e
        delta_vel = desired_vel - self.vel
        return 1 / self.t * delta_vel
    
    def calculate_obstacle_force(self):
        total_f = Vec2(0, 0)
        for ob in all_obstacles:
            for n, d in ob.get_distances(self):
                r = self.r
                f = self.A_ob * exp((r - d) / self.B_ob) * n
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
                r = self.r + ped.r
                f = self.A_ped * exp((r - d) / self.B_ped) * n
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
        # heatmap[int(self.pos.y / g.grid), int(self.pos.x / g.grid)] += 1
    
    def draw(self):
        color = self.color
        if self.waiting:
            color = self.waiting_color
        pygame.draw.aacircle(WIN, color, self.pos, self.r)
        pygame.draw.aacircle(WIN, BLACK, self.pos, self.r, 1)
        #
        m = 3
        pygame.draw.line(WIN, (0, 255, 0), self.pos, self.pos + self.vel * m, 2)
        pygame.draw.line(WIN, (255, 140, 0), self.pos, self.pos + self.acc * m * 7, 2)
        pygame.draw.line(WIN, pygame.Color("brown"), self.pos, self.pos + (self.dest - self.pos).normalize() * m * 4, 2)
        w = "v" if self.follow_vectors else "d"
        WIN.blit(font.render(w, True, (0, 0, 0)), self.pos)
    

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
        pass
    
    @property
    def xy(self):
        return Vec2(self.rect.center)
    
    def get_distances(self, other):
        normal = (other.pos - self.xy)
        return [(normal, normal.length())]
    

class Polygon(AbstractObstacle):
    def __init__(self, name, points, connect=False, invisible=False):
        self.toml_attrs = ("points", "connect", "invisible")
        self.name = name
        self.points = [Vec2(*p) for p in points]
        self.connect = connect
        self.invisible = invisible
        if self.connect:
            self.points.append(self.points[0])
    
    def update(self):
        pass
    
    def draw(self):
        pygame.draw.lines(WIN, BLACK, False, self.points, 1 if self.invisible else 5)
    
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
    def __init__(self, name, p1, n, l, av):
        self.toml_attrs = ("p1", "n", "l", "av")
        self.name = name
        self.p1 = Vec2(p1)
        self.n = n
        self.l = l
        self.av = av
        self.angle = 0
        self.p2s = [Vec2(0, 0) for _ in range(self.n)]
    
    def update(self):
        pass
    
    def draw(self):
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
    def __init__(self, name, area, dimensions, wait_mode="pv", wait=None, kill=False, children=None, chances=None, queue=False, queue_positions=None, queue_initiator=None, code=None):
        self.toml_attrs = ("area", "dimensions", "wait_mode", "wait", "kill", "children", "chances", "queue_positions","queue_initiator", "code")
        self.name = name
        self.area = area
        self.rect = pygame.Rect(area)

        self.dimensions = dimensions
        self.num_x, self.num_y = self.dimensions
        #
        self.attractors = []
        for y in range(self.num_y):
            for x in range(self.num_x):
                center = (self.rect.x + x / self.num_x * self.rect.width + g.grid / 2, self.rect.y + y / self.num_y * self.rect.height + g.grid / 2)
                self.attractors.append(center)
        self.attractor_waiting_data = [[] for _ in self.attractors]
        self.available_attractor_index = 0
        #
        self.children = children if children is not None else []
        self.chances = chances if chances is not None else [1]
        self.last_time = ticks()
        self.pedestrians = []
        self.kill = kill
        self.get_wait = parse_wait(wait)
        self.wait_mode = wait_mode
        self.wait = wait
        self.queue_positions = queue_positions
        self.queue_initiator = queue_initiator
        self.code = code

        if self.wait_mode == "queue":
            self.attractor_rects = [pygame.Rect(pos[0] * g.grid, pos[1] * g.grid, g.grid, g.grid) for pos in self.queue_positions]
            self.attractors = [rect.center for rect in self.attractor_rects]
            self.queue_initiator_rect = self.attractor_rects[self.queue_initiator]

    def get_num_available_attractors(self):
        return sum([not bool(data) for data in self.attractor_waiting_data])

    def get_available_attractor(self):
        try:
            index = random.choice([i for i, data in enumerate(self.attractor_waiting_data) if not data])
        except IndexError:
            index = None
        finally:
            return index
    
    def release_ped(self, ped):
        if self.wait_mode == "att":
            self.attractor_waiting_data[ped.att_index].remove(ped)
        self.pedestrians.remove(ped)
    
    def assign_queue_pos(self, ped):
        pass

    def new_ped(self, ped):
        ped.area = self
        ped.pving = False
        if self.wait_mode == "queue":
            ped.dest = self.queue_initiator_rect.center
            ped.follow_vectors = False
        else:
            ped.follow_vectors = False

        if self.wait_mode == "att":
            if self.attractors:
                ped.att_index = self.get_available_attractor()
                if ped.att_index is not None:
                    ped.area = self
                    ped.dest = self.attractors[ped.att_index]
            else:
                ped.dest = self.rect.center
            self.attractor_waiting_data[ped.att_index].append(ped)

        elif self.wait_mode == "pv":
            ped.dest = self.rect.center

        ped.waiting = False
        ped.wait = self.get_wait()
        self.pedestrians.append(ped)

    def draw(self):
        if self.wait_mode == "queue":
            pygame.draw.rect(WIN, pygame.Color("orange"), self.queue_initiator_rect, 3)
            pygame.draw.rect(WIN, pygame.Color("green"), self.attractor_rects[0], 3)
            pygame.draw.rect(WIN, pygame.Color("CYAN"), self.attractor_rects[self.available_attractor_index], 3)
            WIN.blit(font.render(str(len(self.pedestrians)), True, (0, 0, 255)), self.queue_initiator_rect.center)
        else:
            pygame.draw.rect(WIN, (170, 170, 170), self.area)
            pygame.draw.rect(WIN, (100, 100, 100), self.area, 3)
            for pos in self.attractors:
                pygame.draw.circle(WIN, pygame.Color("#B2D3C2"), pos, 5)
            WIN.blit(font.render(str(len(self.pedestrians)), True, (0, 0, 255)), self.rect.center)

    def update(self):
        self.draw()
        i = 0
        for ped in self.pedestrians.copy():
            i += 1
            ped.update(dt=1)
            if self.wait_mode == "queue":
                # is the pedestrian going towards initiator?
                if ped.dest == self.queue_initiator_rect.center:
                    # did the pedestrian collide with it?
                    if self.queue_initiator_rect.collidepoint(ped.pos):
                        ped.follow_vectors = True
                        ped.dest = self.attractor_rects.pop(0).center
                # did the pedestrian get to the front of the queue?
                pass
            # does pedestrian need to start waiting at the area attractor?
            elif self.wait_mode == "att":
                if (ped.dest - ped.pos).length() <= 1.8 or ped.area.rect.collidepoint(ped.pos):
                    if self.kill:
                        self.pedestrians.remove(ped)
                        all_pedestrians.remove(ped)
                    else:
                        ped.start_waiting()
            elif self.wait_mode == "pv":
                if self.rect.collidepoint(ped.pos):
                    if self.kill:
                        self.pedestrians.remove(ped)
                        all_pedestrians.remove(ped)
                    else:
                        ped.start_pv()
            # did pedestrian wait long enough?
            if self.children and ped.waiting:
                if ticks() - ped.last_wait >= ped.wait:
                    self.release_ped(ped)
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
vector_field = {}
g = None
pool = {}
g = Global("global", 120, 810, 810)
vector_image = pygame.transform.flip(pygame.transform.scale(pygame.image.load(Path("res", "arrow.png")), (g.grid * 0.5, g.grid * 0.5)), False, False)
palette_image = pygame.image.load(Path("res", "palette.png"))
palette = [palette_image.get_at((x, 0)) for x in range(palette_image.width)][1:]


model_path = Path("src", "old_schiphol.toml")
load_model(model_path)

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
heatmap = np.zeros((g.height // g.grid, g.width // g.grid))

g.main()