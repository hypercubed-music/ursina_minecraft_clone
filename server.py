import configparser
import itertools
import os.path
import pickle
import random
import sys
from os import path, makedirs
from pynoise.noisemodule import *
from pynoise.noiseutil import *

import numpy as np
from noise import *
from ursinanetworking import *

Server = UrsinaNetworkingServer("localhost", 25565)

blockIDs = dict()
deferredBlocks = dict()
changedChunks = list()

PRE_GENERATE_DISTANCE = 2
CHUNK_WIDTH = 16
CHUNK_HEIGHT = 256
try:
    world_folder = "worlds\\" + sys.argv[1]
except NameError:
    world_folder = "worlds\\world"
#offsets = [random.randint(-65535, 65535) for i in range(5)]

noise1 = Perlin(octaves=1)
noise2 = Perlin(octaves=5, frequency=1/128)
noise3 = Perlin(octaves=5, frequency=1/128)
noise4 = Perlin(octaves=1, frequency=1/512)

players = dict()
numPlayers = 0


@Server.event
def onClientConnected(Client):
    global numPlayers
    print("Client connected")
    numPlayers += 1
    Client.send_message("recvSessionId", numPlayers)
    Server.broadcast("allPositions", players)
    players[numPlayers] = list()


@Server.event
def onClientDisconnected(Client):
    global numPlayers
    players.pop(numPlayers, None)
    numPlayers -= 1


@Server.event
def posUpdate(Client, Content):
    playerID = Content[0]
    position = Content[1]
    players[playerID] = position
    Server.broadcast("posUpdate", [playerID, position])


@Server.event
def unloadChunk(Client, Content):
    chunk = Content
    chunk_file = str(int(chunk[0])) + "," + str(int(chunk[1])) + "chunk.p"
    if chunk in changedChunks:
        pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunk_file, "wb"))
        changedChunks.remove(chunk)


def encode_chunk(pos):
    pass


def fillCube(id, pos, chunk, xRange=0, yRange=0, zRange=0):
    for x in range(pos[0], pos[0] + xRange + 1):
        for y in range(pos[1], pos[1] + yRange + 1):
            for z in range(pos[2], pos[2] + zRange + 1):
                _pos = [x, y, z]
                _chunk = [chunk[0], chunk[1]]
                if x > CHUNK_WIDTH:
                    _pos[0] -= CHUNK_WIDTH
                    _chunk[0] += 1
                elif x < 1:
                    _pos[0] += CHUNK_WIDTH
                    _chunk[0] -= 1
                if z > CHUNK_WIDTH:
                    _pos[2] -= CHUNK_WIDTH
                    _chunk[1] += 1
                elif z < 1:
                    _pos[2] += CHUNK_WIDTH
                    _chunk[1] -= 1
                _chunk = tuple(_chunk)
                if _chunk in blockIDs:
                    blockIDs[_chunk][_pos[0], _pos[1], _pos[2]] = id
                else:
                    if not _chunk in deferredBlocks:
                        deferredBlocks[_chunk] = list()
                    deferredBlocks[_chunk].append([_pos[0], _pos[1], _pos[2], id])


def fillSphere(id, pos, chunk, rad):
    centerX = rad + pos[0]
    centerY = rad + pos[1]
    centerZ = rad + pos[2]
    for x in range(pos[0], pos[0] + (rad * 2) + 1):
        for y in range(pos[1] - rad, pos[1] + (rad * 2) + 1):
            for z in range(pos[2] - rad, pos[2] + (rad * 2) + 1):
                dx = x - centerX
                dy = y - centerY
                dz = z - centerZ
                if dx ** 2 + dy ** 2 + dz ** 2 < rad ** 2:
                    _pos = [x, y, z]
                    _chunk = [chunk[0], chunk[1]]
                    if x > CHUNK_WIDTH:
                        _pos[0] -= CHUNK_WIDTH
                        _chunk[0] += 1
                    elif x < 1:
                        _pos[0] += CHUNK_WIDTH
                        _chunk[0] -= 1
                    if z > CHUNK_WIDTH:
                        _pos[2] -= CHUNK_WIDTH
                        _chunk[1] += 1
                    elif z < 1:
                        _pos[2] += CHUNK_WIDTH
                        _chunk[1] -= 1
                    _chunk = tuple(_chunk)
                    if _chunk in blockIDs:
                        blockIDs[_chunk][_pos[0], _pos[1], _pos[2]] = id
                    else:
                        if not _chunk in deferredBlocks:
                            deferredBlocks[_chunk] = list()
                        deferredBlocks[_chunk].append([_pos[0], _pos[1], _pos[2], id])


def genTree(chunk, pos):
    height = random.randint(3, 6)
    # fillCube(4, (pos[0]-2, pos[1] + height - 1, pos[2]-2), chunk, 4, 3, 4)
    fillSphere(5, (pos[0] - 3, pos[1] + height - 1, pos[2] - 3), chunk, 3)
    fillCube(4, (pos[0], pos[1], pos[2]), chunk, 0, height, 0)
    # fillCube(random.randint(5, 20), (pos[0], pos[1], pos[2]), chunk, 4, 4, 4)


def _generate(position, chunkWidth, chunkHeight):
    '''
    Generate method used internally for pre-generating
    :return:
    '''
    chunk_file = str(int(position[0])) + "," + str(int(position[2])) + "chunk.p"
    chunk = (position[0], position[2])
    cave_gradient = np.array([(abs(i) * 4 / CHUNK_HEIGHT) / 2 for i in
                              range(-int(CHUNK_HEIGHT), int(CHUNK_HEIGHT * 0.1))])[:256]

    start_x = (position[0] * chunkWidth) - 1
    start_z = (position[2] * chunkWidth) - 1
    end_x = start_x + chunkWidth + 2
    end_z = start_z + chunkWidth + 2
    # opencl gpu go brrrrrrrrrr
    lownoise_map = noise2.get_values(chunkWidth+2, chunkWidth+2, start_x, end_x, start_z, end_z, 0)
    lownoise_map = lownoise_map.reshape((chunkWidth+2), (chunkWidth+2)) * 16 + 16

    highnoise_map = noise3.get_values(chunkWidth+2, chunkWidth+2, start_x, end_x, start_z, end_z, 0)
    highnoise_map = highnoise_map.reshape((chunkWidth+2), (chunkWidth+2))
    #highnoise_map += noise4.get_values(chunkWidth+2, chunkWidth+2, start_x, end_x, start_z, end_z, 0).reshape((chunkWidth+2), (chunkWidth+2))*6
    highnoise_map = highnoise_map * 75 + 16

    cave_map = np.zeros((CHUNK_WIDTH+2, CHUNK_HEIGHT, CHUNK_WIDTH+2))
    for i in range(CHUNK_WIDTH+2):
        noise_map_level = noise1.get_values(CHUNK_WIDTH+2, CHUNK_HEIGHT, start_z/16, end_z/16, 0, 16, (i+start_x)/16, None)
        noise_map_level = (0.5 - np.abs(noise_map_level.reshape((CHUNK_HEIGHT, CHUNK_WIDTH+2)))) * cave_gradient[:, None]
        cave_map[i] = noise_map_level

    #cavenoise_map = noise1.get_values((CHUNK_WIDTH+2)**2, CHUNK_HEIGHT, start_x, end_x, start_z, end_z, 0, CHUNK_HEIGHT*50)
    #cavenoise_map = cavenoise_map.reshape((CHUNK_WIDTH+2, CHUNK_WIDTH+2, CHUNK_HEIGHT))

    if not (position[0], position[2]) in blockIDs:
        if path.exists(world_folder + '\\' + chunk_file) and chunk not in changedChunks:
            blockIDs[(position[0], position[2])] = pickle.load(open(world_folder + '\\' + chunk_file, "rb"))
        else:
            blockIDs[(position[0], position[2])] = np.zeros((chunkWidth + 2, chunkHeight, chunkWidth + 2),
                                                            dtype='uint8')

            for i in itertools.product(range(chunkWidth + 2), range(chunkWidth + 2)):
                blockHeight = int(max(lownoise_map[i[1], i[0]], highnoise_map[i[1], i[0]])) + 32
                # blockHeight = int(highnoise_map[i[1], i[0]])
                # blockHeight = 48
                for y in range(blockHeight):
                    if cave_map[i[0], y, i[1]] > 0.02:
                        blockIDs[chunk][i[0], y, i[1]] = 3 if y == blockHeight - 1 else (
                                2 if (y > blockHeight - 4) else 1)
                if blockHeight < 48:
                    if random.random() > 0.995 and 0 < i[0] < chunkWidth and 0 < i[1] < chunkWidth:
                        genTree(chunk, (i[0], blockHeight, i[1]))
                blockIDs[chunk][i[0], 0, i[1]] = 1

            if chunk in deferredBlocks:
                for i in deferredBlocks[chunk]:
                    blockIDs[chunk][i[0], i[1], i[2]] = i[3]
                deferredBlocks.pop(chunk, None)

            if not path.exists(world_folder):
                makedirs(world_folder)
            #pickle.dump(blockIDs[(position[0], position[2])], open(world_folder + '\\' + chunk_file, "wb"))


@Server.event
def generate(Client, Content):
    '''
    Generate function as called by clients
    :param Client: de client
    :param Content: de content
    :return:
    '''
    position = Content[0]
    chunkWidth = Content[1]
    chunkHeight = Content[2]
    _generate(position, chunkWidth, chunkHeight)
    Client.send_message("recvChunkBlocks", ((position[0], position[2]), blockIDs[(position[0], position[2])]))


@Server.event
def addBlock(Client, Content):
    '''
    add block
    :param Client: de client
    :param Content: de content
    :return:
    '''
    blockPos = Content[0]
    blockID = Content[1]
    chunk = Content[2]
    # chunk_file = str(int(chunk[0])) + "," + str(int(chunk[2])) + "chunk.p"
    blockIDs[chunk][blockPos[0], blockPos[1], blockPos[2]] = blockID
    Server.broadcast("recvChunkBlocks", (chunk, blockIDs[chunk]))
    if chunk not in changedChunks:
        changedChunks.append(chunk)
    # pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunk_file, "wb"))


@Server.event
def deleteBlock(Client, Content):
    '''
    delete block
    :param Client: de client
    :param Content: de content
    :return:
    '''
    blockPos = Content[0]
    chunk = Content[1]
    # chunk_file = str(int(chunk[0])) + "," + str(int(chunk[2])) + "chunk.p"
    blockIDs[chunk][blockPos[0], blockPos[1], blockPos[2]] = 0
    Server.broadcast("serverPrint", (blockPos[0], blockPos[1], blockPos[2]))
    Server.broadcast("recvChunkBlocks", (chunk, blockIDs[chunk]))
    if chunk not in changedChunks:
        changedChunks.append(chunk)
    # pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunk_file, "wb"))


@Server.event
def sendPreGenProgress(prog, total):
    Server.broadcast("preGenProgress", [prog, total])


if __name__ == "__main__":
    print("Hello")
    if os.path.isdir(world_folder):
        # Loading world
        print("Loading world...")
        config = configparser.ConfigParser()
        config.read(world_folder + "\\world.properties")
        seed = config.getint("world", "seed")
        random.seed(seed)
    else:
        # Creating new world
        print("Creating world...")
        seed = random.randrange(sys.maxsize)
        random.seed(seed)
        makedirs(world_folder)
        prop_file = open(world_folder + "\\world.properties", "w")
        prop_file.write("[world]\n")
        prop_file.write("seed=" + str(seed) + "\n")
        prop_file.close()
    noise1.seed = seed
    noise2.seed = seed
    noise3.seed = seed
    noise4.seed = seed
    print("Pre-generating blocks")
    idx = 1
    for i in itertools.product(range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE + 1),
                               range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE + 1)):
        print(idx, "/", (PRE_GENERATE_DISTANCE * 2 + 1) ** 2)
        _generate((i[0], 0, i[1]), CHUNK_WIDTH, CHUNK_HEIGHT)
        # sendPreGenProgress(idx, (PRE_GENERATE_DISTANCE*2+1)**2)
        idx += 1
    print("Done")
    while True:
        try:
            Server.process_net_events()
        except KeyboardInterrupt:
            exit(0)
