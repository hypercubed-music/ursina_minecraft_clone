from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from ursina.entity import Entity
from noise import *
import itertools
import numpy as np
import threading
import random
import time
from tqdm import tqdm
from ursina.scripts.project_uvs import project_uvs
from ursina.shaders import *

CHUNK_WIDTH = 8
CHUNK_HEIGHT = 256
RENDER_DISTANCE = 4
BLOCK_TYPES = 2

'''noise1 = PerlinNoise(octaves=3, seed=random.randint(1, 65535))
noise2 = PerlinNoise(octaves=3, seed=random.randint(1, 65535))
noise3 = PerlinNoise(octaves=8, seed=random.randint(1, 65535))
noise4 = PerlinNoise(octaves=8, seed=random.randint(1, 65535))
noise5 = PerlinNoise(octaves=5, seed=random.randint(1, 65535))
noise6 = PerlinNoise(octaves=8, seed=random.randint(1, 65535))'''
seeds = [random.randint(1,1000) for i in range(10)]
app = Ursina()
coords = Text(text="", origin=(0.75,0.75), background=True)

load_model('block')

blockTex = load_texture('assets/all_blocks.png')
texOffsets = [None, (0,1), (0,0), (1,0), (1,1)]
texFaceOffsets = [[], [(3,0), (3,0), (2,0), (11,1), (3,0), (3,0)],
                  [(19,0) for x in range(6)]]
#TEXIMGWIDTH = 2
TEXIMGHEIGHT = 16
TEXIMGWIDTH = 2

base_verts = [(1, 0, 1), (1, 1, 1), (1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1),#right
              (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0),#back
              (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0),#left
              (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1),#front
              (1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0),#bottom
              (0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (0, 1, 1), (0, 1, 0)]#top
'''top_face = [(0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (0, 1, 1), (0, 1, 0)]
bottom_face = [(1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0)]
left_face = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0)]
right_face = [(1, 0, 1), (1, 1, 1), (1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1)]
front_face = [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)]
back_face = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)]
top_uv = []
bottom_uv = []
left_uv = []
right_uv = []
front_uv = []
back_uv = []'''

base_uvs = [(0.375, 0.0), (0.625, 0.0), (0.625, 0.25), (0.625, 0.25), (0.375, 0.25), (0.375, 0.0),
            (0.375, 0.25),(0.625, 0.25), (0.625, 0.5), (0.625, 0.5), (0.375, 0.5), (0.375, 0.25),
            (0.375, 0.5), (0.625, 0.5),(0.625, 0.75), (0.625, 0.75), (0.375, 0.75), (0.375, 0.5),
            (0.375, 0.75), (0.625, 0.75), (0.625, 1.0),(0.625, 1.0), (0.375, 1.0), (0.375, 0.75),
            (0.125, 0.5), (0.375, 0.5), (0.375, 0.75), (0.375, 0.75),(0.125, 0.75), (0.125, 0.5),
            (0.625, 0.5), (0.875, 0.5), (0.875, 0.75), (0.875, 0.75), (0.625, 0.75),(0.625, 0.5)]

base_norms = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0),
              (-0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
              (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0),
              (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0),
              (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, 1.0, 0.0),
              (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0)]

fpc = FirstPersonController(x=0, y=256, z=0, height=1.7, jump_duration=0.2, jump_height=1.2)
blockHighlight = Entity(model='cube', visible=False, position=(0,0,0), scale=1.1, color=color.rgba(255,255,255,128), origin=(0.45, 0.45, 0.45))

#fpc = EditorCamera(rotation_smoothing=2, enabled=1,rotation_speed = 200, panning_speed=200)
#fpc.gizmo.enabled = True
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
        # self.renderFaces = list()
        self.blockIDs = np.zeros((CHUNK_WIDTH+2, CHUNK_HEIGHT, CHUNK_WIDTH+2), dtype='int16')
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
        mask = (self.blockIDs == 0)
        # black magic (https://stackoverflow.com/questions/68322118/)
        out = np.zeros(mask.shape, dtype='bool')
        out[:-1] = out[:-1] | mask[1:]
        out[1:] = out[1:] | mask[:-1]
        out[:, :-1] = out[:, :-1] | mask[:, 1:]
        out[:, 1:] = out[:, 1:] | mask[:, :-1]
        out[:, :, :-1] = out[:, :, :-1] | mask[:, :, 1:]
        out[:, :, 1:] = out[:, :, 1:] | mask[:, :, :-1]

        temp = np.argwhere(out & (self.blockIDs != 0))
        self.renderBlocks = temp[~np.any(np.logical_or(temp == 0, temp == CHUNK_WIDTH + 1), axis=1)]


    def generate(self):
        global addedBlocks, deletedBlocks
        maxHeight = 0
        time1 = time.perf_counter()
        self.blockIDs = np.zeros((CHUNK_WIDTH+2, CHUNK_HEIGHT, CHUNK_WIDTH+2), dtype='int16')
        gradient = [(i * 4) / CHUNK_HEIGHT for i in range(int(CHUNK_HEIGHT / 2), -int(CHUNK_HEIGHT / 2), -1)]
        for i in itertools.product(range(CHUNK_WIDTH+2), range(CHUNK_WIDTH+2)):
            xpos = (i[0] + (self.position[0]*CHUNK_WIDTH)-1)/100
            zpos = (i[1] + (self.position[2]*CHUNK_WIDTH)-1)/100
            noiseVal = snoise2(x=xpos + snoise2(x=xpos, y=zpos, octaves=3), y=zpos, octaves=3, base=seeds[0])
            blockHeight = math.floor((noiseVal)*15)+48
            if maxHeight < blockHeight:
                maxHeight = (blockHeight if blockHeight < CHUNK_HEIGHT else CHUNK_HEIGHT)
            for y in range(blockHeight if blockHeight < CHUNK_HEIGHT else CHUNK_HEIGHT):
                '''caveNoise = (snoise3(xpos, y, zpos, octaves=6) + (2.0 - (gradient[y]))) * 10
                #caveNoise = 1 if caveNoise > 0 else 0
                if caveNoise > 0:
                    if y <= blockHeight - 3:
                        self.blockIDs[i[0], y, i[1]] = 1
                    else:
                        self.blockIDs[i[0], y, i[1]] = 2'''
                if y <= blockHeight - 3:
                    self.blockIDs[i[0], y, i[1]] = 1
                else:
                    self.blockIDs[i[0], y, i[1]] = 2
            self.blockIDs[i[0], 0, i[1]] = 1
        '''gradient = [(i*4) / CHUNK_HEIGHT for i in range(int(CHUNK_HEIGHT/2), -int(CHUNK_HEIGHT/2), -1)]
        for i in itertools.product(range(CHUNK_WIDTH + 2), range(CHUNK_WIDTH + 2)):
            xpos = (i[0] + (self.position[0] * CHUNK_WIDTH) - 1)
            zpos = (i[1] + (self.position[2] * CHUNK_WIDTH) - 1)
            for y in range(CHUNK_HEIGHT):
                heightNoise = (snoise3(xpos / 128 + snoise3(xpos / 8192, y / 8192, zpos / 8192, octaves=1), y / 8192,
                                       zpos / 128, octaves=6) +
                               gradient[y]) * 10
                heightNoise = 1 if heightNoise > 0 else 0
                caveNoise = (snoise3(xpos / 64, y / 64, zpos / 64, octaves=6) + (1.5 - (gradient[y]))) * 10
                caveNoise = 1 if caveNoise > 0 else 0
                self.blockIDs[i[0], y, i[1]] = int(heightNoise if caveNoise == 1 else 0)
            self.blockIDs[i[0], 0, i[1]] = 1'''
        #maxHeight = max(np.argmax(self.blockIDs,axis=1))
        # Generate added/removed blocks
        if (self.position[0], self.position[2]) in addedBlocks:
            for added in addedBlocks[(self.position[0], self.position[2])]:
                self.blockIDs[added[0], added[1], added[2]] = added[3]
        if (self.position[0], self.position[2]) in deletedBlocks:
            for deleted in deletedBlocks[(self.position[0], self.position[2])]:
                self.blockIDs[deleted[0], deleted[1], deleted[2]] = 0
        # get blocks we need to actually render
        self.getRenderable()
        print("generate: " + str(time.perf_counter() - time1))
        self.isGenerated = True

    def render(self):
        self.unrender()
        self.updateBorder()
        i = 0
        # print(self.renderFaces)
        for block in self.renderBlocks:
            # self.addToMesh((block + self.position*(CHUNK_WIDTH-1)), self.renderFaces[i], int(self.blockIDs[block[0], block[1], block[2]]))

            self.addToMesh((block + self.position * (CHUNK_WIDTH - 1)), int(self.blockIDs[block[0], block[1], block[2]]))
            i += 1
        if self.verts is None:
            self.isRendered = True
            return
        self.model = Mesh(vertices=[tuple(i) for i in self.verts.tolist()], normals=self.norms, uvs=self.uvs)
        self.texture = blockTex
        self.collider = MeshCollider(self, mesh=self.model, center=Vec3(0,0,0))
        self.visible_self = True
        self.isRendered = True
        # self.shader = lit_with_shadows_shader
        print(len(self.verts))

    def unrender(self):
        self.model = None
        self.verts = None
        self.uvs = list()
        self.norms = list()

    def addToMesh(self, pos, blockID=1):
        '''texPos = texFaceOffsets[blockID]
        face_verts = [right_face, left_face, top_face, bottom_face, front_face, back_face]
        face_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0,1.0), (0.0,1.0), (0.0, 0.0)]
        face_norms = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        _pos = np.array(pos)
        for idx, face in enumerate(faces):
            if face:
                if self.verts is None:
                    self.verts = np.array(face_verts[idx]) + _pos

                else:
                    self.verts = np.append(self.verts, base_verts + _pos, axis=0)
                self.uvs += [((i[0] + texPos[idx][0])/TEXIMGWIDTH, (i[1] + texPos[idx][1])/TEXIMGHEIGHT) for i in face_uvs]
                self.norms += [face_norms[idx] for i in range(6)]'''
        _pos = np.array(pos)
        if self.verts is None:
            self.verts = np.array(base_verts) + _pos
        else:
            self.verts = np.append(self.verts, base_verts + _pos, axis=0)
        self.uvs += [((i[0] + texOffsets[blockID][0])/TEXIMGWIDTH, (i[1] + texOffsets[blockID][1])/TEXIMGWIDTH) for i in base_uvs]
        self.norms += base_norms

    def deleteBlock(self, position):
        global deletedBlocks
        _position = [int(position.x), int(position.y), int(position.z)]
        if self.blockIDs[_position[0],_position[1], _position[2]] != 0:
            self.blockIDs[_position[0],_position[1], _position[2]] = 0
        removearray(self.renderBlocks, _position)
        if ((self.position[0], self.position[2]) in addedBlocks):
            deletedBlocks[(self.position[0], self.position[2])].append(tuple(_position))
        else:
            deletedBlocks[(self.position[0], self.position[2])] = [tuple(_position)]
        # update surrounding blocks
        self.checkSurrounding(_position)
        self.render()

    def addBlock(self, position, id):
        global addedBlocks
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
            if (self.position[0], self.position[2]) in addedBlocks:
                addedBlocks[(self.position[0], self.position[2])].append((_position[0], _position[1], _position[2], id))
            else:
                addedBlocks[(self.position[0], self.position[2])] = [(_position[0], _position[1], _position[2], id)]
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
            if not arreq_in_list(np.array(newpos), ch.renderBlocks):
                ch.checkRenderable(tuple(newpos))
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

def doChunkRendering(_currentChunk):
    # unload distant chunks
    xRange = range(_currentChunk[0] - RENDER_DISTANCE, _currentChunk[0] + RENDER_DISTANCE + 1)
    zRange = range(_currentChunk[1] - RENDER_DISTANCE, _currentChunk[1] + RENDER_DISTANCE + 1)
    for idx, chunk in enumerate(renderedChunkPos):
        if not (chunk[0] in xRange and chunk[1] in zRange):
            destroy(renderedChunks[idx])
            del renderedChunks[idx]
            del renderedChunkPos[idx]
            break

    # load chunks in range (one at a time)
    if (not len(renderedChunks) == 0) and (not renderedChunks[-1].isRendered) and renderedChunks[-1].isGenerated:
        #threading.Thread(renderedChunks[-1].render(), daemon=True).start()
        renderedChunks[-1].render()
    else:
        for i in itertools.product(xRange, zRange):
            if not (i[0], i[1]) in renderedChunkPos:
                renderedChunkPos.append((i[0], i[1]))
                renderedChunks.append(Chunk((i[0], i[1])))
                #threading.Thread(renderedChunks[-1].generate(), daemon=True).start()
                renderedChunks[-1].generate()
                break

    if not renderedChunks[-1].hasCollider:
        renderedChunks[-1].setCollider()

def input(key):
    global mouseChunk, lookingAt
    if key == "left mouse down" and lookingAt is not None:
        getChunk(mouseChunk).deleteBlock(Vec3(lookingAt.x - (CHUNK_WIDTH * mouseChunk[0]), lookingAt.y, lookingAt.z - (CHUNK_WIDTH * mouseChunk[1])))
    elif key == "right mouse down" and lookingAt is not None:
        getChunk(mouseChunk).addBlock(Vec3(lookingAt.x - ((CHUNK_WIDTH) * mouseChunk[0]) + mouse.normal.x,
                                           lookingAt.y + mouse.normal.y,
                                           lookingAt.z - ((CHUNK_WIDTH) * mouseChunk[1]) + mouse.normal.z), 1)

chunkThread = threading.Thread()
def update():
    global currentChunk, lookingAt, mouseChunk, chunkThread
    currentChunk = (math.floor(fpc.position[0] / CHUNK_WIDTH), math.floor(fpc.position[2] / CHUNK_WIDTH))
    #currentChunk = doChunkRendering()
    if chunkThread is None:
        chunkThread = threading.Thread(doChunkRendering(currentChunk)).start()
    elif not chunkThread.is_alive():
        chunkThread = threading.Thread(doChunkRendering(currentChunk)).start()
    if mouse.hovered_entity is not None:
        blockHighlight.visible = True
        # chunk offset for some reason
        lookingAt = Vec3(math.floor(mouse.world_point.x - 0.5*mouse.normal.x),
                         math.floor(mouse.world_point.y - 0.5*mouse.normal.y),
                         math.floor(mouse.world_point.z - 0.5*mouse.normal.z))
        blockHighlight.position = lookingAt + Vec3(1, 1, 1)
        mouseChunk = (math.floor((lookingAt[0]-1) / CHUNK_WIDTH), math.floor((lookingAt[2]-1) / CHUNK_WIDTH))
    else:
        blockHighlight.visible = False
        lookingAt = None

    if held_keys["control"]:
        fpc.speed = 10
        camera.fov = 110
    else:
        fpc.speed = 5
        camera.fov = 90

    coords.text = ", ".join([str(int(i)) for i in list(fpc.position)]) + "\n" + (str(list(lookingAt)) if lookingAt is not None else "")

sky = Sky(color="87ceeb", texture=None)
#DirectionalLight(y=2, z=3, shadows=True, rotation_z = 45)
while len(renderedChunks) < math.pow(RENDER_DISTANCE * 2 + 1, 2):
    doChunkRendering((0,0))
    print(len(renderedChunks), "/", math.pow(RENDER_DISTANCE * 2 + 1, 2))
print(max(getChunk((0,0)).blockIDs[0, :, 0].argsort()))
fpc.y = max(np.argwhere(getChunk((0,0)).blockIDs[1, :, 1] != 0)) + 10
app.run()