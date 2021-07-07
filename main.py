from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.entity import Entity
from perlin_noise import PerlinNoise
import itertools
import numpy as np
import threading
import random

from ursina.scripts.project_uvs import project_uvs

CHUNK_WIDTH = 8
CHUNK_HEIGHT = 32
RENDER_DISTANCE = 3
BLOCK_TYPES = 2

noise1 = PerlinNoise(octaves=1, seed=random.randint(1, 65535))
noise2 = PerlinNoise(octaves=5, seed=random.randint(1, 65535))
app = Ursina()
coords = Text(text="", origin=(0.75,0.75), background=True)

load_model('block')

blockTex = load_texture('assets/all_blocks.png')
texOffsets = [None, (0,1), (0,0), (1,0), (1,1)]
TEXIMGWIDTH = 2

base_verts = [(1, 0, 1), (1, 1, 1), (1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1),
              (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0),
              (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0),
              (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1),
              (1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0),
              (0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (0, 1, 1), (0, 1, 0)]


base_uvs = [(0.375, 0.0), (0.625, 0.0), (0.625, 0.25), (0.625, 0.25), (0.375, 0.25), (0.375, 0.0), (0.375, 0.25),
            (0.625, 0.25), (0.625, 0.5), (0.625, 0.5), (0.375, 0.5), (0.375, 0.25), (0.375, 0.5), (0.625, 0.5),
            (0.625, 0.75), (0.625, 0.75), (0.375, 0.75), (0.375, 0.5), (0.375, 0.75), (0.625, 0.75), (0.625, 1.0),
            (0.625, 1.0), (0.375, 1.0), (0.375, 0.75), (0.125, 0.5), (0.375, 0.5), (0.375, 0.75), (0.375, 0.75),
            (0.125, 0.75), (0.125, 0.5), (0.625, 0.5), (0.875, 0.5), (0.875, 0.75), (0.875, 0.75), (0.625, 0.75),
            (0.625, 0.5)]
base_norms = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0),
              (-0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
              (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0),
              (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0),
              (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, 1.0, 0.0),
              (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0)]

fpc = FirstPersonController(x=0, y=20, z=0, height=0.85)
blockHighlight = Entity(model='cube', visible=False, position=(0,0,0), scale=1.1, color=color.rgba(255,255,255,128), origin=(0.45, 0.45, 0.45))

#fpc = EditorCamera()
renderedChunks = []
renderedChunkPos = []
currentChunk = list()
mouseChunk = list()
lookingAt = Vec3()
addedBlocks = dict()
deletedBlocks = dict()

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

class Chunk(Entity):
    def __init__(self, position=(0,0), seed=0):
        super().__init__(visible_self=False, position=(position[0]*CHUNK_WIDTH, 0, position[1]*CHUNK_WIDTH))
        self.renderBlocks = list()
        self.blockIDs = np.zeros((CHUNK_WIDTH+2, 32, CHUNK_WIDTH+2), dtype='int8')
        self.position = np.array([position[0], 0, position[1]])
        self.position_xz = position
        self.isGenerated = False
        self.isRendered = False
        self.hasCollider = False
        self.verts = None
        self.uvs = list()
        self.norms = list()

    def getRenderable(self, maxHeight=(CHUNK_HEIGHT-1)):
        # Get a list of renderable blocks
        self.renderBlocks.clear()
        coords = np.array(
            [i for i in itertools.product(range(1, CHUNK_WIDTH + 1), range(1, maxHeight + 1), range(1, CHUNK_WIDTH + 1))
             if self.blockIDs[i] != 0], dtype='int8')
        for i in coords:
            if self.checkRenderable(tuple(i)):
                self.renderBlocks.append(i)

    def generate(self):
        maxHeight = 0
        self.blockIDs = np.zeros((CHUNK_WIDTH+2, 32, CHUNK_WIDTH+2))
        for i in itertools.product(range(CHUNK_WIDTH+2), range(CHUNK_WIDTH+2)):
            xpos = (i[0] + (self.position[0]*CHUNK_WIDTH)-1)/100
            zpos = (i[1] + (self.position[2]*CHUNK_WIDTH)-1)/100
            noiseVal = noise1([xpos, zpos]) + 0.2 * noise2([xpos, zpos])
            blockHeight = math.floor((noiseVal)*30)+15
            if maxHeight < blockHeight:
                maxHeight = (blockHeight if blockHeight < 32 else 32)
            for y in range(blockHeight if blockHeight < 32 else 32):
                if y <= blockHeight - 3:
                    self.blockIDs[i[0], y, i[1]] = 1
                else:
                    self.blockIDs[i[0], y, i[1]] = 2
        # get blocks we need to actually render
        self.getRenderable(maxHeight)
        self.isGenerated = True

    def checkRenderable(self, pos):
        _pos = [int(i) for i in pos]
        if self.blockIDs[_pos[0], _pos[1], _pos[2]] == 0:
            return False
        surround_list = [self.blockIDs[_pos[0] + 1, _pos[1], _pos[2]], self.blockIDs[_pos[0] - 1, _pos[1], _pos[2]],
                             self.blockIDs[_pos[0], _pos[1] + 1, _pos[2]], self.blockIDs[_pos[0], _pos[1] - 1, _pos[2]],
                             self.blockIDs[_pos[0], _pos[1], _pos[2] + 1], self.blockIDs[_pos[0], _pos[1], _pos[2] - 1]]
        return 0 in surround_list

    def render(self):
        self.unrender()
        self.updateBorder()
        for block in self.renderBlocks:
            self.addToMesh((block + self.position*(CHUNK_WIDTH-1)), int(self.blockIDs[block[0], block[1], block[2]]))
        if self.verts is None:
            return
        self.model = Mesh(vertices=[tuple(i) for i in self.verts.tolist()], normals=self.norms, uvs=self.uvs)
        self.texture = blockTex
        self.collider = MeshCollider(self, mesh=self.model, center=Vec3(0,0,0))
        self.visible_self = True
        self.isRendered = True

    def unrender(self):
        #for block in self.blocks:
        #    destroy(block)
        self.model = None
        self.verts = None
        self.uvs = list()
        self.norms = list()

    def addToMesh(self, pos, blockID=1):
        _pos = np.array(pos)
        if self.verts is None:
            self.verts = np.array(base_verts) + _pos
        else:
            self.verts = np.append(self.verts, base_verts + _pos, axis=0)
        self.uvs += [((i[0] + texOffsets[blockID][0])/TEXIMGWIDTH, (i[1] + texOffsets[blockID][1])/TEXIMGWIDTH) for i in base_uvs]
        self.norms += base_norms

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
        if _position[0] > CHUNK_WIDTH:
            getChunk((self.position[0] + 1, self.position[2])).addBlock(Vec3(_position[0], _position[1], _position[2], ) - Vec3(CHUNK_WIDTH, 0, 0), id)
        if _position[0] < 1:
            getChunk((self.position[0] - 1, self.position[2])).addBlock(Vec3(_position[0], _position[1], _position[2], ) + Vec3(CHUNK_WIDTH, 0, 0), id)
        if _position[2] > CHUNK_WIDTH:
            getChunk((self.position[0], self.position[2] + 1)).addBlock(Vec3(_position[0], _position[1], _position[2], ) - Vec3(0, 0, CHUNK_WIDTH), id)
        if _position[2] < 1:
            getChunk((self.position[0], self.position[2] - 1)).addBlock(Vec3(_position[0], _position[1], _position[2], ) + Vec3(0, 0, CHUNK_WIDTH), id)
        if self.blockIDs[_position[0], _position[1], _position[2]] != id:
            self.blockIDs[_position[0], _position[1], _position[2]] = id
        self.checkSurrounding(_position)
        self.render()

    def checkSurrounding(self, pos):
        chList = []
        for i in itertools.product(range(pos[0]-1, pos[0]+2), range(pos[1]-1, pos[1]+2), range(pos[2]-1, pos[2]+2)):
            ch = self
            chPos = [self.position[0], self.position[2]]
            newpos = list(i)
            if i[0] > CHUNK_WIDTH:
                # check adjacent chunk
                chPos[0] += 1
                newpos[0] -= CHUNK_WIDTH
            if i[0] < 1:
                chPos[0] -= 1
                newpos[0] += CHUNK_WIDTH
            if i[2] > CHUNK_WIDTH:
                chPos[1] += 1
                newpos[2] -= CHUNK_WIDTH
            if i[2] < 1:
                chPos[1] -= 1
                newpos[2] += CHUNK_WIDTH
            if chPos != [self.position[0], self.position[2]]:
                ch = getChunk(tuple(chPos))
                if ch not in chList and ch != self:
                    chList.append(ch)
            if ch.checkRenderable(newpos) and not arreq_in_list(np.array(newpos), ch.renderBlocks):
                ch.renderBlocks.append(np.array(newpos))
                ch.render()
        for ch in chList:
            ch.render()

    def setCollider(self):
        self.collider = MeshCollider(self, mesh=self.model, center=Vec3(0, 0, 0))
        self.hasCollider = True

    def updateBorder(self):
        global renderedChunkPos
        if (self.position[0]+1, self.position[2]) in renderedChunkPos:
            self.blockIDs[CHUNK_WIDTH+1,:,:] = getChunk((self.position[0]+1, self.position[2])).blockIDs[1,:,:]
        if (self.position[0]-1, self.position[2]) in renderedChunkPos:
            self.blockIDs[0,:,:] = getChunk((self.position[0]-1, self.position[2])).blockIDs[CHUNK_WIDTH,:,:]
        if (self.position[0], self.position[2]+1) in renderedChunkPos:
            self.blockIDs[:,:,CHUNK_WIDTH+1] = getChunk((self.position[0], self.position[2]+1)).blockIDs[:,:,1]
        if (self.position[0], self.position[2]-1) in renderedChunkPos:
            self.blockIDs[:,:,0] = getChunk((self.position[0], self.position[2]-1)).blockIDs[:,:,CHUNK_WIDTH]
        self.getRenderable()

#fpc = EditorCamera(rotation_speed = 200, panning_speed=200)

def doChunkRendering():
    currentChunk = (math.floor(fpc.position[0] / CHUNK_WIDTH), math.floor(fpc.position[2] / CHUNK_WIDTH))
    # unload distant chunks
    xRange = range(currentChunk[0] - RENDER_DISTANCE, currentChunk[0] + RENDER_DISTANCE + 1)
    zRange = range(currentChunk[1] - RENDER_DISTANCE, currentChunk[1] + RENDER_DISTANCE + 1)
    for idx, chunk in enumerate(renderedChunkPos):
        if not (chunk[0] in xRange and chunk[1] in zRange):
            print("delete " + str(chunk))
            destroy(renderedChunks[idx])
            del renderedChunks[idx]
            del renderedChunkPos[idx]
            break

    # load chunks in range (one at a time)
    if (not len(renderedChunks) == 0) and (not renderedChunks[-1].isRendered):
        threading.Thread(renderedChunks[-1].render(), daemon=True).start()
    else:
        for i in itertools.product(xRange, zRange):
            if not (i[0], i[1]) in renderedChunkPos:
                renderedChunkPos.append((i[0], i[1]))
                renderedChunks.append(Chunk((i[0], i[1])))
                threading.Thread(renderedChunks[-1].generate(), daemon=True).start()
                print("render at " + str(i))
                break

    if not renderedChunks[-1].hasCollider:
        renderedChunks[-1].setCollider()
    return currentChunk

def input(key):
    global mouseChunk, lookingAt
    if key == "left mouse down" and lookingAt is not None:
        getChunk(mouseChunk).deleteBlock(Vec3(lookingAt.x - (CHUNK_WIDTH * mouseChunk[0]), lookingAt.y, lookingAt.z - (CHUNK_WIDTH * mouseChunk[1])))
    elif key == "right mouse down" and lookingAt is not None:
        getChunk(mouseChunk).addBlock(Vec3(lookingAt.x - ((CHUNK_WIDTH) * mouseChunk[0]) + mouse.normal.x,
                                           lookingAt.y + mouse.normal.y,
                                           lookingAt.z - ((CHUNK_WIDTH) * mouseChunk[1]) + mouse.normal.z), 1)
def update():
    global currentChunk, lookingAt, mouseChunk
    #if fpc.grounded:
    #    currentChunk = doChunkRendering()
    currentChunk = doChunkRendering()

    if mouse.point is not None:
        blockHighlight.visible = True
        # chunk offset for some reason
        lookingAt = Vec3(math.floor(mouse.world_point.x - 0.5*mouse.normal.x),
                         math.floor(mouse.world_point.y - 0.5*mouse.normal.y),
                         math.floor(mouse.world_point.z - 0.5*mouse.normal.z))
        blockHighlight.position = lookingAt + Vec3(1, 1, 1)
        mouseChunk = (math.floor((lookingAt[0]-1) / CHUNK_WIDTH), math.floor((lookingAt[2]-1) / CHUNK_WIDTH))
    else:
        lookingAt = None

    coords.text = ", ".join([str(int(i)) for i in list(fpc.position)]) + "\n" + (str(list(lookingAt)) if lookingAt is not None else "")

sky = Sky(color="87ceeb", texture=None)
AmbientLight(y=2, z=3, shadows=True)
while len(renderedChunks) < math.pow(RENDER_DISTANCE * 2 + 1, 2):
    doChunkRendering()
app.run()