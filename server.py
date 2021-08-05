import configparser
import itertools
import os.path
import pickle
import random
import sys
from os import path, makedirs
from pynoise.noisemodule import *
from pynoise.noiseutil import *
from threading import Thread

import numpy as np
from ursinanetworking import *

Server = UrsinaNetworkingServer("localhost", 25565)

blockIDs = dict()
deferredBlocks = dict()
changedChunks = list()

PRE_GENERATE_DISTANCE = 2
CHUNK_WIDTH = 16
CHUNK_HEIGHT = 16

try:
    world_folder = "worlds\\" + sys.argv[1]
except NameError:
    world_folder = "worlds\\world"
# offsets = [random.randint(-65535, 65535) for i in range(5)]

noise1 = Perlin(octaves=1)
noise2 = Perlin(octaves=5, frequency=1 / 128)
noise3 = Perlin(octaves=5, frequency=1 / 128)
noise4 = Perlin(octaves=1, frequency=1 / 512)

players = dict()
numPlayers = 0


@Server.event
def onClientConnected(Client):
    global numPlayers
    print("Client connected")
    numPlayers += 1
    if len(players) != 0:
        newSessionID = max(players.keys()) + 1
    else:
        newSessionID = 1
    Client.send_message("recvSessionId", newSessionID)
    Server.broadcast("allPositions", players)
    players[newSessionID] = list()


''''@Server.event
def onClientDisconnected(Client):
    global numPlayers
    players.pop(numPlayers, None)
    numPlayers -= 1'''


@Server.event
def playerQuit(Client, sessionID):
    global numPlayers
    players.pop(sessionID, None)
    numPlayers -= 1
    if numPlayers == 0:
        print("No more players. Saving all chunks...")
        for chunk in changedChunks:
            chunkFile = str(int(chunk[0])) + "," + str(int(chunk[2])) + "chunk.p"
            # pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunkFile, "wb"))
            changedChunks.remove(chunk)


@Server.event
def posUpdate(Client, Content):
    playerID = Content[0]
    position = Content[1]
    players[playerID] = position
    Server.broadcast("posUpdate", [playerID, position])


@Server.event
def unloadChunk(Client, Content):
    chunk = Content
    chunk_file = str(int(chunk[0])) + "," + str(int(chunk[1])) + "," + str(int(chunk[2])) + "chunk.p"
    if chunk in changedChunks:
        # pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunk_file, "wb"))
        changedChunks.remove(chunk)
    print("Saving unloaded chunk")


def fillCube(id, pos, chunk, xRange=0, yRange=0, zRange=0):
    for x in range(pos[0], pos[0] + xRange + 1):
        for y in range(pos[1], pos[1] + yRange + 1):
            for z in range(pos[2], pos[2] + zRange + 1):
                # _pos = [x, y, z]
                # _chunk = [chunk[0], chunk[1], chunk[2]]
                _pos = [x % CHUNK_WIDTH + 1, y % CHUNK_HEIGHT + 1, z % CHUNK_WIDTH + 1]
                _chunk = [chunk[0] + (x // CHUNK_WIDTH),
                          chunk[1] + (y // CHUNK_HEIGHT),
                          chunk[2] + (z // CHUNK_WIDTH)]
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
                    _pos = [x % CHUNK_WIDTH + 1, y % CHUNK_HEIGHT + 1, z % CHUNK_WIDTH + 1]
                    _chunk = [chunk[0] + (x // CHUNK_WIDTH),
                              chunk[1] + (y // CHUNK_HEIGHT),
                              chunk[2] + (z // CHUNK_WIDTH)]
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
    Generates a single chunk at <position>
    :return:
    '''
    chunk_file = str(int(position[0])) + "," + str(int(position[1])) + "," + str(int(position[2])) + "chunk.p"
    chunk = (position[0], position[1], position[2])
    cave_gradient = np.array([abs(i) / 128 for i in
                              range(128, 0, -1)])

    start_x = (position[0] * chunkWidth) - 1
    start_y = (position[1] * chunkHeight) - 1
    start_z = (position[2] * chunkWidth) - 1
    end_x = start_x + chunkWidth + 2
    end_y = start_y + chunkHeight + 2
    end_z = start_z + chunkWidth + 2
    # opencl gpu go brrrrrrrrrr
    lownoise_map = noise2.get_values(chunkWidth + 2, chunkWidth + 2, start_x, end_x, start_z, end_z, 0)
    lownoise_map = lownoise_map.reshape((chunkWidth + 2), (chunkWidth + 2)) * 16 + 16

    highnoise_map = noise3.get_values(chunkWidth + 2, chunkWidth + 2, start_x, end_x, start_z, end_z, 0)
    highnoise_map = highnoise_map.reshape((chunkWidth + 2), (chunkWidth + 2))
    # highnoise_map += noise4.get_values(chunkWidth+2, chunkWidth+2, start_x, end_x, start_z, end_z, 0).reshape((chunkWidth+2), (chunkWidth+2))*6
    highnoise_map = highnoise_map * 75 + 16

    cave_map = np.zeros((CHUNK_WIDTH + 2, CHUNK_HEIGHT + 2, CHUNK_WIDTH + 2))
    for i in range(CHUNK_WIDTH + 2):
        # noise_map_level = noise1.get_values(CHUNK_WIDTH+2, CHUNK_HEIGHT, start_z/16, end_z/16, 0, 16, (i+start_x)/16, None)
        noise_map_level = noise1.get_values(CHUNK_WIDTH + 2, CHUNK_HEIGHT + 2, start_z / 8, end_z / 8, start_y / 8, end_y / 8,
                                            (i + start_x) / 8, None)
        noise_map_level = 0.5 - np.abs(noise_map_level.reshape((CHUNK_HEIGHT + 2, CHUNK_WIDTH + 2)))
        cave_gradient = np.array([max(min((i+64)/128, 1.0), 0.1) for i in range(start_y, end_y)])
        noise_map_level = noise_map_level * cave_gradient[:, None]
        '''if start_y+64 > 0 and end_y+64 < len(cave_gradient):
            noise_map_level = noise_map_level * cave_gradient[start_y+64:end_y+64, None]
        elif start_y+64 > len(cave_gradient):
            noise_map_level = noise_map_level * 0'''
        cave_map[i] = noise_map_level

    # cavenoise_map = noise1.get_values((CHUNK_WIDTH+2)**2, CHUNK_HEIGHT, start_x, end_x, start_z, end_z, 0, CHUNK_HEIGHT*50)
    # cavenoise_map = cavenoise_map.reshape((CHUNK_WIDTH+2, CHUNK_WIDTH+2, CHUNK_HEIGHT))

    if not (position[0], position[1], position[2]) in blockIDs:
        if path.exists(world_folder + '\\' + chunk_file) and chunk not in changedChunks:
            blockIDs[(position[0], position[1], position[2])] = pickle.load(
                open(world_folder + '\\' + chunk_file, "rb"))
        else:
            blockIDs[(position[0], position[1], position[2])] = np.zeros(
                (chunkWidth + 2, chunkHeight + 2, chunkWidth + 2),
                dtype='uint8')
            for i in itertools.product(range(chunkWidth + 2), range(chunkWidth + 2)):
                blockHeight = int(max(lownoise_map[i[1], i[0]], highnoise_map[i[1], i[0]])) + 32
                if blockHeight < position[1] * chunkHeight:
                    continue
                for y in range(min(chunkHeight + 2, (blockHeight - start_y) + 1)):

                    absolute_y = y + start_y - 1
                    if cave_map[i[0], y, i[1]] > 0.01:
                    # if cave_map[i[0], y, i[1]] > 0.02:
                        blockIDs[chunk][i[0], y, i[1]] = 3 if absolute_y == blockHeight - 1 else (
                            2 if (absolute_y > blockHeight - 4) else 1)
                if blockHeight < 48 and start_y < blockHeight < end_y:
                    if random.random() > 0.995 and 0 < i[0] < chunkWidth and 0 < i[1] < chunkWidth:
                        genTree(chunk, (i[0], blockHeight - start_y, i[1]))

            if chunk in deferredBlocks:
                for i in deferredBlocks[chunk]:
                    blockIDs[chunk][i[0], i[1], i[2]] = i[3]
                deferredBlocks.pop(chunk, None)

            if not path.exists(world_folder):
                makedirs(world_folder)
            # pickle.dump(blockIDs[(position[0],position[1], position[2])], open(world_folder + '\\' + chunk_file, "wb"))


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
    isEmpty = np.all(blockIDs[(position[0], position[1], position[2])] == 0)
    if isEmpty:
        Client.send_message("recvChunkBlocks", ((position[0], position[1], position[2]), None, isEmpty))
    else:
        Client.send_message("recvChunkBlocks", (
            (position[0], position[1], position[2]), blockIDs[(position[0], position[1], position[2])], isEmpty))


def updateAdjacentChunks(blockPos, Client, chunk):
    if blockPos[0] == 1:
        Client.send_message("recvChunkBlocks", (
            (chunk[0] - 1, chunk[1], chunk[2]),
            None if np.all(blockIDs[chunk] == 0) else blockIDs[(chunk[0] - 1, chunk[1], chunk[2])],
            np.all(blockIDs[chunk] == 0)))
    if blockPos[0] == CHUNK_WIDTH:
        Client.send_message("recvChunkBlocks", (
            (chunk[0] + 1, chunk[1], chunk[2]),
            None if np.all(blockIDs[chunk] == 0) else blockIDs[(chunk[0] + 1, chunk[1], chunk[2])],
            np.all(blockIDs[chunk] == 0)))
    if blockPos[1] == 1:
        Client.send_message("recvChunkBlocks", (
            (chunk[0], chunk[1] - 1, chunk[2]),
            None if np.all(blockIDs[chunk] == 0) else blockIDs[(chunk[0], chunk[1] - 1, chunk[2])],
            np.all(blockIDs[chunk] == 0)))
    if blockPos[1] == CHUNK_WIDTH:
        Client.send_message("recvChunkBlocks", (
            (chunk[0], chunk[1] + 1, chunk[2]),
            None if np.all(blockIDs[chunk] == 0) else blockIDs[(chunk[0], chunk[1] + 1, chunk[2])],
            np.all(blockIDs[chunk] == 0)))
    if blockPos[2] == 1:
        Client.send_message("recvChunkBlocks", (
            (chunk[0], chunk[1], chunk[2] - 1),
            None if np.all(blockIDs[chunk] == 0) else blockIDs[(chunk[0], chunk[1], chunk[2] - 1)],
            np.all(blockIDs[chunk] == 0)))
    if blockPos[2] == CHUNK_WIDTH:
        Client.send_message("recvChunkBlocks", (
            (chunk[0], chunk[1], chunk[2] + 1),
            None if np.all(blockIDs[chunk] == 0) else blockIDs[(chunk[0], chunk[1], chunk[2] + 1)],
            np.all(blockIDs[chunk] == 0)))


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
    # Server.broadcast("recvChunkBlocks", (chunk, blockIDs[chunk]))
    Client.send_message("recvChunkBlocks", (
        chunk, None if np.all(blockIDs[chunk] == 0) else blockIDs[chunk], np.all(blockIDs[chunk] == 0)))
    if chunk not in changedChunks:
        changedChunks.append(chunk)
    updateAdjacentChunks(blockPos, Client, chunk)
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
    # Server.broadcast("serverPrint", (blockPos[0], blockPos[1], blockPos[2]))
    # Server.broadcast("recvChunkBlocks", (chunk, blockIDs[chunk]))
    isEmpty = np.all(blockIDs[chunk] == 0)
    if isEmpty:
        Client.send_message("recvChunkBlocks", (chunk, None, isEmpty))
    else:
        Client.send_message("recvChunkBlocks", (chunk, blockIDs[chunk], isEmpty))
    updateAdjacentChunks(blockPos, Client, chunk)
    if chunk not in changedChunks:
        changedChunks.append(chunk)
    # pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunk_file, "wb"))


@Server.event
def stop(Client, Content):
    print("Saving all chunks...")
    for chunk in changedChunks:
        chunkFile = str(int(chunk[0])) + "," + str(int(chunk[2])) + "chunk.p"
        # pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunkFile, "wb"))
        changedChunks.remove(chunk)
    exit(0)


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
        seed = random.randrange((2 ** 31 - 1))
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
                               range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE + 1),
                               range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE + 1)):
        print(idx, "/", (PRE_GENERATE_DISTANCE * 2 + 1) ** 3)
        _generate((i[0], i[1], i[2]), CHUNK_WIDTH, CHUNK_HEIGHT)
        # sendPreGenProgress(idx, (PRE_GENERATE_DISTANCE*2+1)**2)
        idx += 1
    print("Done")
    while True:
        try:
            Server.process_net_events()
        except KeyboardInterrupt:
            exit(0)
