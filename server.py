import itertools
import random
from noise import *
import numpy as np
from ursinanetworking import *
import pickle
from os import path, makedirs

Server = UrsinaNetworkingServer("localhost", 25565)

blockIDs = dict()
deferredBlocks = dict()

PRE_GENERATE_DISTANCE = 2
CHUNK_WIDTH = 16
CHUNK_HEIGHT = 256
world_folder = "world"
generateNew = True
offsets = [random.randint(-65535,65535) for i in range(5)]
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
    print("position update")
    Server.broadcast("posUpdate", [playerID, position])

def fillCube(id, pos, chunk, xRange=0, yRange=0, zRange=0):
    for x in range(pos[0], pos[0] + xRange+1):
        for y in range(pos[1], pos[1] + yRange+1):
            for z in range(pos[2], pos[2] + zRange+1):
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
                    deferredBlocks[_chunk].append([_pos[0],_pos[1],_pos[2], id])

def fillSphere(id, pos, chunk, rad):
    centerX = rad + pos[0]
    centerY = rad + pos[1]
    centerZ = rad + pos[2]
    for x in range(pos[0], pos[0]+(rad*2)+1):
        for y in range(pos[1]-rad, pos[1]+(rad*2)+1):
            for z in range(pos[2]-rad, pos[2]+(rad*2)+1):
                dx = x - centerX
                dy = y - centerY
                dz = z - centerZ
                if dx**2 + dy**2 + dz**2 < rad**2:
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
    height = random.randint(3,6)
    #fillCube(4, (pos[0]-2, pos[1] + height - 1, pos[2]-2), chunk, 4, 3, 4)
    fillSphere(4, (pos[0]-3, pos[1] + height - 1, pos[2]-3), chunk, 3)
    fillCube(3, (pos[0], pos[1], pos[2]), chunk, 0, height, 0)
    #fillCube(random.randint(5, 20), (pos[0], pos[1], pos[2]), chunk, 4, 4, 4)

def _generate(position, chunkWidth, chunkHeight):
    '''
    Generate method used internally for pre-generating
    :return:
    '''
    chunk_file = str(int(position[0])) + "," + str(int(position[2])) + "chunk.p"
    chunk = (position[0], position[2])
    cave_gradient = [(-2 + abs(i) * 4 / chunkHeight) / 2 for i in
                     range(int(chunkHeight * 0.1), -int(chunkHeight), -1)]
    if not (position[0], position[2]) in blockIDs:
        if path.exists(world_folder + '\\' + chunk_file) and not generateNew:
            blockIDs[(position[0], position[2])] = pickle.load(open(world_folder + '\\' + chunk_file, "rb"))
        else:
            blockIDs[(position[0], position[2])] = np.zeros((chunkWidth + 2, chunkHeight, chunkWidth + 2), dtype='int16')

            @np.vectorize
            def caveNoiseGen(x, y, z):
                x += offsets[0]
                z += offsets[1]
                return snoise3(x / 32, y / 32, z / 32, octaves=2) + (1.4 + (cave_gradient[y])) > 0

            @np.vectorize
            def heightNoiseGen(x, z):
                x += offsets[2]
                z += offsets[3]
                lowNoise = snoise2(x / 256, z / 256, octaves=5) * 32 + 76
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
                if blockHeight < 32:
                    if random.random() > 0.995 and 0 < i[0] < chunkWidth and 0 < i[1] < chunkWidth:
                        genTree(chunk, (i[0], blockHeight, i[1]))
                for y in range(blockHeight):
                    blockIDs[chunk][i[0], y, i[1]] = (2 if (y > blockHeight - 3) else 1) if caveNoise[y, i[0], i[1]] == 1 else 0
                blockIDs[chunk][i[0], 0, i[1]] = 1

            if chunk in deferredBlocks:
                for i in deferredBlocks[chunk]:
                    blockIDs[chunk][i[0], i[1], i[2]] = i[3]
                deferredBlocks.pop(chunk, None)

            if not path.exists(world_folder):
                makedirs(world_folder)
            pickle.dump(blockIDs[(position[0], position[2])], open(world_folder + '\\' + chunk_file, "wb"))

@Server.event
def generate(Client, Content):
    '''
    Generate function as called by clients
    :param Client: de client
    :param Content: de content
    :return:
    '''
    position = Content[0]
    print("Recieved chunk request ", str(position))
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
    #chunk_file = str(int(chunk[0])) + "," + str(int(chunk[2])) + "chunk.p"
    blockIDs[chunk][blockPos[0], blockPos[1], blockPos[2]] = blockID
    Server.broadcast("recvChunkBlocks", (chunk, blockIDs[chunk]))
    #pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunk_file, "wb"))

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
    #chunk_file = str(int(chunk[0])) + "," + str(int(chunk[2])) + "chunk.p"
    blockIDs[chunk][blockPos[0], blockPos[1], blockPos[2]] = 0
    Server.broadcast("recvChunkBlocks", (chunk, blockIDs[chunk]))
    #pickle.dump(blockIDs[chunk], open(world_folder + '\\' + chunk_file, "wb"))

@Server.event
def sendPreGenProgress(prog, total):
    Server.broadcast("preGenProgress", [prog, total])

if __name__ == "__main__":
    print("Hello")
    print("Pre-generating blocks")
    idx = 1
    for i in itertools.product(range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE+1),
                               range(-PRE_GENERATE_DISTANCE, PRE_GENERATE_DISTANCE+1)):
        print(idx, "/", (PRE_GENERATE_DISTANCE*2+1)**2)
        _generate((i[0], 0, i[1]), CHUNK_WIDTH, CHUNK_HEIGHT)
        #sendPreGenProgress(idx, (PRE_GENERATE_DISTANCE*2+1)**2)
        idx += 1
    print("Done")
    try:
        while True:
            Server.process_net_events()
    except KeyboardInterrupt:
        exit()