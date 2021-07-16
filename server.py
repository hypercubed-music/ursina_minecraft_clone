import itertools
from noise import *
import numpy as np
from ursinanetworking import *

Server = UrsinaNetworkingServer("localhost", 25565)

added_blocks = dict()
deleted_blocks = dict()
blockIDs = dict()

PRE_GENERATE_DISTANCE = 5
CHUNK_WIDTH = 16
CHUNK_HEIGHT = 256

@Server.event
def onClientConnected(Client):
    print("Client connected")

def _generate(position, chunkWidth, chunkHeight):
    '''
    Generate method used internally for pre-generating
    :return:
    '''
    cave_gradient = [(-2 + abs(i) * 4 / chunkHeight) / 2 for i in
                     range(int(chunkHeight * 0.1), -int(chunkHeight), -1)]
    if not (position[0], position[2]) in blockIDs:
        blockIDs[(position[0], position[2])] = np.zeros((chunkWidth + 2, chunkHeight, chunkWidth + 2), dtype='int16')

        @np.vectorize
        def caveNoiseGen(x, y, z):
            return snoise3(x / 32, y / 32, z / 32, octaves=2) + (1.4 + (cave_gradient[y])) > 0

        @np.vectorize
        def heightNoiseGen(x, z):
            lowNoise = snoise2(x / 128, z / 128, octaves=5) * 32 + 76
            highNoise = ((snoise2(x / 64, z / 64, octaves=5) + snoise2(x / 1024, z / 1024,
                                                                       octaves=1)) * 6) * 5 + 72
            return max(lowNoise, highNoise) - 32

        x, y, z = np.meshgrid(np.arange(chunkWidth + 2) + (position[0] * chunkWidth) - 1,
                              np.arange(chunkHeight),
                              np.arange(chunkWidth + 2) + (position[2] * chunkWidth) - 1)
        x2, z2 = np.meshgrid(np.arange(chunkWidth + 2) + (position[0] * chunkWidth) - 1,
                             np.arange(chunkWidth + 2) + (position[2] * chunkWidth) - 1)
        heightNoise = heightNoiseGen(x2, z2)
        caveNoise = caveNoiseGen(x, y, z)
        for i in itertools.product(range(chunkWidth + 2), range(chunkWidth + 2)):
            blockHeight = int(heightNoise[i[1], i[0]])
            for y in range(blockHeight):
                blockIDs[(position[0], position[2])][i[0], y, i[1]] = (2 if (y > blockHeight - 3) else 1) if caveNoise[y, i[0], i[1]] == 1 else 0
            blockIDs[(position[0], position[2])][i[0], 0, i[1]] = 1

@Server.event
def generate(Client, Content):
    position = Content[0]
    chunkWidth = Content[1]
    chunkHeight = Content[2]
    _generate(position, chunkWidth, chunkHeight)
    Client.send_message("recvChunkBlocks", ((position[0], position[2]), blockIDs[(position[0], position[2])]))

@Server.event
def addBlock(Client, Content):
    blockPos = Content[0]
    blockID = Content[1]
    chunk = Content[2]
    blockIDs[chunk][blockPos[0], blockPos[1], blockPos[2]] = blockID
    Client.send_message("recvChunkBlocks", (chunk, blockIDs[chunk]))

@Server.event
def deleteBlock(Client, Content):
    blockPos = Content[0]
    chunk = Content[1]
    blockIDs[chunk][blockPos[0], blockPos[1], blockPos[2]] = 0
    Client.send_message("recvChunkBlocks", (chunk, blockIDs[chunk]))

def sendPreGenProgress(prog, total):
    Server.broadcast("preGenProgress", [prog, total])

if __name__ == "__main__":
    print("Hello")
    print("Pre-generating blocks")
    i = 0
    for i in itertools.product(range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE+1),
                               range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE+1)):
        print(i, "/", (PRE_GENERATE_DISTANCE*2+1)**2)
        _generate((i[0], 0, i[1]), CHUNK_WIDTH, CHUNK_HEIGHT)
    try:
        while True:
            Server.process_net_events()
    except KeyboardInterrupt:
        exit()