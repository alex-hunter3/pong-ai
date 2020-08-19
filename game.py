import neat
import pygame
import os

from sys import exit
from time import time
from pickle import dump
from random import choice, randint


pygame.init()


class Player:
    def __init__(self, counter):
        self.width      = 20
        self.height     = 70
        self.y          = 215
        self.score      = 0
        self.velocity   = 8
        self.x          = 20          if counter == 0 else win_width - self.width * 2
        self.colour     = (255, 0, 0) if counter == 0 else (0, 0, 255)
        self.identifier = "Left"      if counter == 0 else "Right"

    def draw(self, win, decision):
        self.move(decision)
        pygame.draw.rect(win, self.colour, (self.x, self.y, self.width, self.height), 2)

    def move(self, decision):                                        # this here to take more active neuron, mention in paper
        if decision[1] > 0 and self.y + self.height < win_height and decision[1] > decision[0]:
        	self.y += self.velocity
        elif decision[0] > 0 and self.y > 0 and decision[1] < decision[0]:
        	self.y -= self.velocity

        # if decision[1] > 0 and decision[1] > decision[0]:
        # 	print(f'{self.identifier} player decision: DOWN on {decision}')
        # elif decision[0] > 0 and decision[1] < decision[0]:
        # 	print(f'{self.identifier} player decision: UP, on {decision}')


class Ball:
    def __init__(self):
    	self.x_vel_choices = [8, -8]
    	self.x             = win_width // 2
    	self.y             = win_height // 2
    	self.radius        = 8
    	self.colour        = (255, 255, 255)
    	self.x_vel         = choice(self.x_vel_choices)
    	self.y_vel         = randint(-6, 6)

    	while self.y_vel == 0:
            self.y_vel = randint(-8, 8)

    def draw(self, win):
        self.move()
        pygame.draw.circle(win, self.colour, (self.x, self.y), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel


def main(genomes, config):
    global win_width, win_height, bg

    networks        = []
    network_fitness = []
    players         = []

    hit_count  = 0
    counter    = 0
    win_width  = 800
    win_height = 500
    bg         = (30, 30, 30)
    run        = True

    for _, network in genomes:
        networks.append(neat.nn.FeedForwardNetwork.create(network, config))
        network.fitness = 0
        network_fitness.append(network)
        players.append(Player(counter))
        counter += 1

    del counter

    ball  = Ball()
    clock = pygame.time.Clock()
    win   = pygame.display.set_mode((win_width, win_height), pygame.FULLSCREEN)

    win.fill(bg)
    pygame.display.set_caption("Pong AI")

    while run:
        # clock.tick(60)
        win.fill(bg)

        key = pygame.key.get_pressed()

        if key[pygame.K_ESCAPE]:
            pygame.display.quit()
            quit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                exit()

        for x, player in enumerate(players):
        	inputs = (ball.x_vel, ball.y_vel, ball.x, ball.y - (player.y + player.height // 2))
        	output = networks[x].activate(inputs)
        	player.draw(win, output)

        if ball.y + ball.radius <= 0 or ball.y + ball.radius >= win_height:
            ball.y_vel *= -1

        for player in players:
            if ball.y - ball.radius <= player.y + player.height and ball.y + ball.radius >= player.y:
                if ball.x + ball.radius >= player.x and ball.x - ball.radius <= player.x + player.width:
                    ball.x_vel *= -1
                    hit_count += 1
                    ind = players.index(player)
                    network_fitness[ind].fitness += 5

        if ball.x >= win_width:
            network_fitness[0].fitness += 0.1

            players.pop(1)
            networks.pop(1)
            network_fitness.pop(1)

        elif ball.x <= 0:
            network_fitness[1].fitness += 0.1

            players.pop(0)
            networks.pop(0)
            network_fitness.pop(0)

        ball.draw(win)

        if len(players) == 1 or hit_count == 30:
        	run = False
        	break

        pygame.display.update()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p     = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    start = time()

    winner = p.run(main, 600)
    print(f'\nTime to train model: {round(time() - start, 2)} seconds.')

    with open("Pong Neural Network Model.pkl", "wb") as f:
	    dump(winner, f)
	    pygame.quit()
	    exit()


if __name__ == "__main__":
    local_dir   = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
else:
    print(f"MUST BE RAN IN MAIN FILE {__file__}")
    pygame.quit()
    exit()
