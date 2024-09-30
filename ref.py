import pygame
import random
import math

# Pygame setup
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Constants for forces
GOAL_FORCE = 0.3
PEDESTRIAN_REPULSION = 10000
WALL_REPULSION = 5000
MAX_SPEED = 1
SMALL = 0.0000001

# Exit positions (on the right side)
EXITS = [
    (SCREEN_WIDTH - 50, SCREEN_HEIGHT // 5 * 1),
    (SCREEN_WIDTH - 50, SCREEN_HEIGHT // 5 * 2),
    (SCREEN_WIDTH - 50, SCREEN_HEIGHT // 5 * 3),
    (SCREEN_WIDTH - 50, SCREEN_HEIGHT // 5 * 4),
]

# Pedestrian class
class Pedestrian:
    def __init__(self, x, y, goal):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = 4
        self.goal = goal  # Chosen goal (either EXIT_1 or EXIT_2)
        # self.color = RED if self.goal == EXIT_1 else BLUE
        self.color = BLUE

    def calculate_goal_force(self):
        self.goal = pygame.mouse.get_pos()
        # Force pulling toward the goal (Exit)
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        dist = math.hypot(dx, dy)
        fx = (dx / (dist + SMALL)) * GOAL_FORCE * dist
        fy = (dy / (dist + SMALL)) * GOAL_FORCE * dist
        return fx, fy

    def calculate_repulsion(self, other):
        # Repulsion from other pedestrians
        dx = self.x - other.x
        dy = self.y - other.y
        dist = math.hypot(dx, dy)
        if dist < self.radius * 5:  # Only repulse if too close
            force = PEDESTRIAN_REPULSION / ((dist + SMALL) ** 2)
            fx = (dx / (dist + SMALL)) * force
            fy = (dy / (dist + SMALL)) * force
            return fx, fy
        return 0, 0

    def calculate_wall_repulsion(self):
        # Repulsion from the walls (left and right boundaries)
        fx, fy = 0, 0
        if self.x < 100:  # Left wall
            fx = WALL_REPULSION / (self.x ** 2)
        if self.x > SCREEN_WIDTH - 100:  # Right wall
            fx = -WALL_REPULSION / ((SCREEN_WIDTH - self.x) ** 2)
        if self.y < 100:  # Top wall
            fy = WALL_REPULSION / (self.y ** 2)
        if self.y > SCREEN_HEIGHT - 100:  # Bottom wall
            fy = -WALL_REPULSION / ((SCREEN_HEIGHT - self.y) ** 2)
        return fx, fy

    def move(self, pedestrians):
        # Calculate forces
        goal_fx, goal_fy = self.calculate_goal_force()
        wall_fx, wall_fy = self.calculate_wall_repulsion()

        total_fx, total_fy = goal_fx + wall_fx, goal_fy + wall_fy

        repulse_fx, repulse_fy = 0, 0
        for other in pedestrians:
            if other is not self:
                r = self.calculate_repulsion(other)
                repulse_fx += r[0]
                repulse_fy += r[1]
        total_fx += repulse_fx
        total_fy += repulse_fy

        # Update velocity based on forces
        self.vx += total_fx
        self.vy += total_fy

        # Limit the speed
        speed = math.hypot(self.vx, self.vy)
        if speed > MAX_SPEED:
            self.vx = (self.vx / speed) * MAX_SPEED
            self.vy = (self.vy / speed) * MAX_SPEED

        # Update position
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# Simulation class to manage the environment
class Simulation:
    def __init__(self, num_pedestrians):
        self.pedestrians = [Pedestrian(50, random.gauss(SCREEN_HEIGHT / 2, 200), random.choice(EXITS))
                            for _ in range(num_pedestrians)]
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Social Forces Pedestrian Simulation")

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update pedestrians
            for ped in self.pedestrians:
                ped.move(self.pedestrians)

            # Draw everything
            self.screen.fill(WHITE)

            # Draw exits
            for e in EXITS:
                pygame.draw.circle(self.screen, GREEN, e, 10)

            # Draw pedestrians
            for ped in self.pedestrians:
                ped.draw(self.screen)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

# Main entry point
if __name__ == "__main__":
    sim = Simulation(num_pedestrians=200)
    sim.run()