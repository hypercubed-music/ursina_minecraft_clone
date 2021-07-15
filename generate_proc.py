import itertools
from noise import *
import numpy as np
from ursinanetworking import *

Server = UrsinaNetworkingServer("localhost", 25565)

@Server.event
def onClientConnected(Client):
    print("Client connected")

@Server.event
def onClientDisconnected(Client):
    if Server.clients == 0:
        exit()

@Server.event
def generate(Client, Content):
    position = Content[0]
    chunkWidth = Content[1]
    chunkHeight = Content[2]
    addedBlocks = Content[3]
    deletedBlocks = Content[4]
    cave_gradient = [(-2 + abs(i) * 4 / chunkHeight) / 2 for i in
                     range(int(chunkHeight * 0.1), -int(chunkHeight), -1)]
    blockIDs = np.zeros((chunkWidth + 2, chunkHeight, chunkWidth + 2), dtype='int16')
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
            blockIDs[i[0], y, i[1]] = (2 if (y > blockHeight - 3) else 1) if caveNoise[
                                                                                      y, i[0], i[1]] == 1 else 0
        blockIDs[i[0], 0, i[1]] = 1
    for added in addedBlocks:
        blockIDs[added[0], added[1], added[2]] = added[3]
    for deleted in deletedBlocks:
        blockIDs[deleted[0], deleted[1], deleted[2]] = 0
    #print("generate: " + str(time.perf_counter() - time1))
    #isGenerated = True
    Client.send_message("recvChunkBlocks", (position[0], position[2], blockIDs))

if __name__ == "__main__":
    print("Hello")
    try:
        while True:
            Server.process_net_events()
    except KeyboardInterrupt:
        exit()