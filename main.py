from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from perlin_noise import PerlinNoise
import itertools
import numpy as np
import time
import threading

CHUNK_WIDTH = 8
CHUNK_HEIGHT = 32
RENDER_DISTANCE = 3

noise1 = PerlinNoise(octaves=1)
noise2 = PerlinNoise(octaves=5)
app = Ursina()

load_model('block')

blockTex = [
    None,
    load_texture('assets/grass.png'), # 0
    load_texture('assets/grass.png'), # 1
    load_texture('assets/stone.png'), # 2
    load_texture('assets/gold.png'),  # 3
    load_texture('assets/lava.png'),  # 4
]

fpc = FirstPersonController(x=5, y=20, z=5, height=0.85)
renderedChunks = []
renderedChunkPos = []

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def getChunk(pos):
    if pos in renderedChunkPos:
        return renderedChunks[renderedChunkPos.index(pos)]

class Block(Entity):
    def __init__(self, position=(0,0,0), texture=None, parent=None, render_queue=0):
        super().__init__(
            model='block',
            position=position,
            color=color.white,
            texture=texture,
            scale=0.5,
            collision=True,
            collider='box',
            parent=parent,
            render_queue=render_queue
        )

    def input(self, key):
        if self.hovered:
            if key == 'right mouse down':
                new_position = self.position + mouse.normal - (self.parent.position*(CHUNK_WIDTH-1))
                print(new_position)
                chunk_xz = self.parent.position_xz
                if new_position.x >= CHUNK_WIDTH+1:
                    getChunk((chunk_xz[0] + 1, chunk_xz[1])).addBlock(Vec3(new_position.x-CHUNK_WIDTH, new_position.y, new_position.z), 1)
                elif new_position.x <= 0:
                    getChunk((chunk_xz[0] - 1, chunk_xz[1])).addBlock(Vec3(new_position.x + CHUNK_WIDTH, new_position.y, new_position.z), 1)
                elif new_position.z >= CHUNK_WIDTH+1:
                    getChunk((chunk_xz[0], chunk_xz[1] + 1)).addBlock(Vec3(new_position.x, new_position.y, new_position.z-CHUNK_WIDTH), 1)
                elif new_position.z <= 0:
                    getChunk((chunk_xz[0], chunk_xz[1] - 1)).addBlock(Vec3(new_position.x, new_position.y, new_position.z + CHUNK_WIDTH), 1)
                else:
                    self.parent.addBlock(new_position, 1)
            elif key == 'left mouse down':
                print(self.position - (self.parent.position*(CHUNK_WIDTH-1)))
                self.parent.deleteBlock(self.position - (self.parent.position*(CHUNK_WIDTH-1)))
                destroy(self)

    def update(self):
        if self.hovered:
            self.color = color.gray
        else:
            self.color = color.white

class Chunk(Entity):
    def __init__(self, position=(0,0), seed=0):
        super().__init__(visible_self=False, position=(position[0]*CHUNK_WIDTH, 0, position[1]*CHUNK_WIDTH))
        self.blocks = list()
        self.renderBlocks = list()
        self.blockIDs = np.zeros((CHUNK_WIDTH+2, 32, CHUNK_WIDTH+2), dtype='int8')
        self.position = np.array([position[0], 0, position[1]])
        self.position_xz = position
        self.isGenerated = False
        self.isRendered = False

    def generate(self):
        time1 = time.perf_counter()
        self.blockIDs = np.zeros((CHUNK_WIDTH+2, 32, CHUNK_WIDTH+2))
        for i in itertools.product(range(CHUNK_WIDTH+2), range(CHUNK_WIDTH+2)):
            xpos = (i[0] + (self.position[0]*CHUNK_WIDTH)-1)/100
            zpos = (i[1] + (self.position[2]*CHUNK_WIDTH)-1)/100
            noiseVal = noise1([xpos, zpos]) + 0.2 * noise2([xpos, zpos])
            blockHeight = math.floor((noiseVal)*30)+15
            for y in range(blockHeight if blockHeight < 32 else 32):
                if y <= blockHeight - 3:
                    self.blockIDs[i[0], y, i[1]] = 3
                else:
                    self.blockIDs[i[0], y, i[1]] = 1
        # get blocks we need to actually render
        self.renderBlocks.clear()
        coords = np.array(list(itertools.product(range(1, CHUNK_WIDTH+1), range(1, 31), range(1, CHUNK_WIDTH+1))), dtype='int8')
        for i in coords:
            if self.checkRenderable(tuple(i)):
                self.renderBlocks.append(i)
        print("generate time: ", time.perf_counter() - time1)
        self.isGenerated = True

    def checkRenderable(self, pos):
        _pos = [int(i) for i in pos]
        surround_list = [self.blockIDs[_pos[0] + 1, _pos[1], _pos[2]], self.blockIDs[_pos[0] - 1, _pos[1], _pos[2]],
                             self.blockIDs[_pos[0], _pos[1] + 1, _pos[2]], self.blockIDs[_pos[0], _pos[1] - 1, _pos[2]],
                             self.blockIDs[_pos[0], _pos[1], _pos[2] + 1], self.blockIDs[_pos[0], _pos[1], _pos[2] - 1]]
        return 0 in surround_list and self.blockIDs[pos] != 0

    def render(self):
        self.unrender()
        time2 = time.perf_counter()
        i = 0
        for block in self.renderBlocks:
            i += 1
            self.blocks.append(Block(position=(block + self.position*(CHUNK_WIDTH-1)), texture=blockTex[int(self.blockIDs[block[0], block[1], block[2]])], parent=self, render_queue=i))
        print("render time: ", time.perf_counter() - time2)
        self.isRendered = True

    def unrender(self):
        for block in self.blocks:
            destroy(block)
        self.blocks.clear()

    def deleteBlock(self, position):
        _position = [int(position.x), int(position.y), int(position.z)]
        if self.blockIDs[_position[0],_position[1], _position[2]] != 0:
            self.blockIDs[_position[0],_position[1], _position[2]] = 0
        removearray(self.renderBlocks, _position)
        # update surrounding blocks
        self.checkSurrounding(_position)
        self.render()

    def addBlock(self, position, id):
        _position = [int(position.x), int(position.y), int(position.z)]
        if self.blockIDs[_position[0], _position[1], _position[2]] != id:
            self.blockIDs[_position[0], _position[1], _position[2]] = id
        self.checkSurrounding(_position)
        self.render()

    def checkSurrounding(self, pos):
        for i in itertools.product(range(pos[0]-1, pos[0]+2), range(pos[1]-1, pos[1]+2), range(pos[2]-1, pos[2]+2)):
            if i[0] > CHUNK_WIDTH or i[0] < 1 or i[2] > CHUNK_WIDTH or i[2] < 1:
                continue
            if self.checkRenderable(i) and not arreq_in_list(np.array(i), self.renderBlocks):
                self.renderBlocks.append(np.array(i))

#fpc = EditorCamera(rotation_speed = 200, panning_speed=200)


def update():
    currentChunk = (int(fpc.position[0] // CHUNK_WIDTH), int(fpc.position[2] // CHUNK_WIDTH))
    #unload distant chunks
    xRange = range(currentChunk[0]-RENDER_DISTANCE, currentChunk[0]+RENDER_DISTANCE+1)
    zRange = range(currentChunk[1]-RENDER_DISTANCE, currentChunk[1]+RENDER_DISTANCE+1)
    for idx, chunk in enumerate(renderedChunkPos):
        if not (chunk[0] in xRange and chunk[1] in zRange):
            print("delete " + str(chunk))
            renderedChunks[idx].unrender()
            del renderedChunks[idx]
            del renderedChunkPos[idx]
            return

    #load chunks in range (one at a time)
    if (not len(renderedChunks) == 0) and (not renderedChunks[-1].isRendered):
        threading.Thread(renderedChunks[-1].render()).start()
    else:
        for i in itertools.product(xRange, zRange):
            if not (i[0], i[1]) in renderedChunkPos:
                renderedChunkPos.append((i[0], i[1]))
                renderedChunks.append(Chunk((i[0], i[1])))
                threading.Thread(renderedChunks[-1].generate()).start()
                print("render at " + str(i))
                break

sky = Sky(color="87ceeb", texture=None)
for i in range(162):
    update()
app.run()