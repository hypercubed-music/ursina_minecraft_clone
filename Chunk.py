import random

from ursina import *
import numpy as np
import itertools
from noise import *
from ursina.shaders import *
from multiprocessing import Process, Queue
from ursinanetworking import *

CHUNK_WIDTH = 16
CHUNK_HEIGHT = 256
TEXIMGHEIGHT = 16
TEXIMGWIDTH = 32

blockTex = load_texture('assets/atlas.png')
# texOffsets = [None, (0,1), (0,0), (1,0), (1,1)]
texFaceOffsets = np.array([[(0, 31), (0, 31), (0, 31), (0, 31), (0, 31), (0, 31)], [(19, 31) for x in range(6)],
                           [(3, 31), (3, 31), (3, 31), (3, 31), (11, 30), (2, 31)]])
seeds = [random.randint(1, 1000) for i in range(10)]
base_verts = np.array([(1, 0, 1), (1, 1, 1), (1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1),  # right
                       (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0),  # back
                       (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0),  # left
                       (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1),  # front
                       (1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0),  # bottom
                       (0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (0, 1, 1), (0, 1, 0)])  # top

base_norms = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0), (-0.0, 0.0, -1.0),
              (-0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
              (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0),
              (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, 0.0, 1.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0),
              (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, -1.0, 0.0), (-0.0, 1.0, 0.0),
              (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0), (-0.0, 1.0, 0.0)]

right_face = np.array([(1, 0, 1), (1, 1, 1), (1, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1)])  # right
back_face = np.array([(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)])  # back
left_face = np.array([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0)])  # left
front_face = np.array([(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)])  # front
bottom_face = np.array([(1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0)])  # bottom
top_face = np.array([(0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (0, 1, 1), (0, 1, 0)])

renderedChunks = []
renderedChunkPos = []
currentChunk = list()
mouseChunk = list()
lookingAt = Vec3()
addedBlocks = dict()
deletedBlocks = dict()

Client = UrsinaNetworkingClient("localhost", 25565)

def getChunk(pos):
    if pos in renderedChunkPos:
        return renderedChunks[renderedChunkPos.index(pos)]


class Chunk(Entity):
    def __init__(self, position=(0, 0), seed=0):
        super().__init__(visible_self=False, position=(position[0] * CHUNK_WIDTH, 0, position[1] * CHUNK_WIDTH))
        self.renderBlocks = list()
        # self.renderFaces = list()
        self.blockIDs = np.zeros((CHUNK_WIDTH + 2, CHUNK_HEIGHT, CHUNK_WIDTH + 2), dtype='int16')
        self.position = np.array([position[0], 0, position[1]])
        self.position_xz = position
        self.isGenerated = False
        self.isRendered = False
        self.hasCollider = False
        self.verts = None
        self.uvs = None
        self.norms = list()
        self.shader = lit_with_shadows_shader
        self.generate_proc = Process()

    def getRenderable(self, maxHeight=(CHUNK_HEIGHT - 1)):
        # Get a list of renderable blocks
        mask = (self.blockIDs == 0)

        right = np.argwhere(mask[1:] & (self.blockIDs[:-1] != 0))
        left = np.argwhere(mask[:-1] & (self.blockIDs[1:] != 0)) + np.array([1, 0, 0])
        top = np.argwhere(mask[:, 1:] & (self.blockIDs[:, :-1] != 0))
        bottom = np.argwhere(mask[:, :-1] & (self.blockIDs[:, 1:] != 0)) + np.array([0, 1, 0])
        front = np.argwhere(mask[:, :, 1:] & (self.blockIDs[:, :, :-1] != 0))
        back = np.argwhere(mask[:, :, :-1] & (self.blockIDs[:, :, 1:] != 0)) + np.array([0, 0, 1])

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
        return right, left, top, bottom, front, back

    def generate(self):
        global addedBlocks, deletedBlocks, Client
        time1 = time.perf_counter()
        if Client.connected:
            _pos = (self.position.x, self.position.y, self.position.z)
            '''if (self.position[0], self.position[2]) in addedBlocks:
                ourAdded = addedBlocks[(self.position[0], self.position[2])]
            else:
                ourAdded = []
            if (self.position[0], self.position[2]) in deletedBlocks:
                ourDeleted = deletedBlocks[(self.position[0], self.position[2])]
            else:
                ourDeleted = []'''
            Client.send_message("generate", [_pos, CHUNK_WIDTH, CHUNK_HEIGHT])

    def render(self):
        time1 = time.perf_counter()
        right, left, top, bottom, front, back = self.getRenderable()
        self.unrender()
        self.updateBorder()
        self.buildMesh(left, right, top, bottom, front, back)
        if self.verts is None:
            self.isRendered = True
            return
        self.model = Mesh(vertices=[tuple(i) for i in self.verts],
                          uvs=self.uvs.tolist())
        self.texture = blockTex
        self.collider = MeshCollider(self, mesh=self.model, center=Vec3(0, 0, 0))
        self.visible_self = True
        self.isRendered = True
        print("render: " + str(time.perf_counter() - time1))

    def unrender(self):
        self.model = None
        self.verts = None
        self.uvs = None
        self.norms = list()

    def buildMesh(self, left, right, top, bottom, front, back):
        face_uvs = np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
        self.verts = np.empty((0, 3))
        self.uvs = np.empty((0, 2))
        self.norms = []
        offset = self.position * (CHUNK_WIDTH - 1)
        left_verts = np.tile(left_face, (len(left), 1)) + np.repeat(left, 6, axis=0) + offset
        right_verts = np.tile(right_face, (len(right), 1)) + np.repeat(right, 6, axis=0) + offset
        top_verts = np.tile(top_face, (len(top), 1)) + np.repeat(top, 6, axis=0) + offset
        bottom_verts = np.tile(bottom_face, (len(bottom), 1)) + np.repeat(bottom, 6, axis=0) + offset
        front_verts = np.tile(front_face, (len(front), 1)) + np.repeat(front, 6, axis=0) + offset
        back_verts = np.tile(back_face, (len(back), 1)) + np.repeat(back, 6, axis=0) + offset
        self.verts = np.concatenate((left_verts, right_verts, top_verts, bottom_verts, front_verts, back_verts))
        left_uvs = ((np.tile(face_uvs, (len(left), 1)) +
                     np.repeat(texFaceOffsets[self.blockIDs[left[:, 0], left[:, 1], left[:, 2]], 0], 6,
                               axis=0))) / TEXIMGWIDTH
        right_uvs = ((np.tile(face_uvs, (len(right), 1)) +
                      np.repeat(texFaceOffsets[self.blockIDs[right[:, 0], right[:, 1], right[:, 2]], 1], 6,
                                axis=0))) / TEXIMGWIDTH
        top_uvs = ((np.tile(face_uvs, (len(top), 1)) +
                    np.repeat(texFaceOffsets[self.blockIDs[top[:, 0], top[:, 1], top[:, 2]], 5], 6,
                              axis=0))) / TEXIMGWIDTH
        bottom_uvs = ((np.tile(face_uvs, (len(bottom), 1)) +
                       np.repeat(texFaceOffsets[self.blockIDs[bottom[:, 0], bottom[:, 1], bottom[:, 2]], 4], 6,
                                 axis=0))) / TEXIMGWIDTH
        front_uvs = ((np.tile(face_uvs, (len(front), 1)) +
                      np.repeat(texFaceOffsets[self.blockIDs[front[:, 0], front[:, 1], front[:, 2]], 2], 6,
                                axis=0))) / TEXIMGWIDTH
        back_uvs = ((np.tile(face_uvs, (len(back), 1)) +
                     np.repeat(texFaceOffsets[self.blockIDs[back[:, 0], back[:, 1], back[:, 2]], 3], 6,
                               axis=0))) / TEXIMGWIDTH
        self.uvs = np.concatenate((left_uvs, right_uvs, top_uvs, bottom_uvs, front_uvs, back_uvs))


    def deleteBlock(self, position):
        global deletedBlocks
        _position = [int(position.x), int(position.y), int(position.z)]
        Client.send_message("deleteBlock", [_position, (self.position[0], self.position[2])])
        # update surrounding blocks
        #self.checkSurrounding(_position)
        #self.render()

    def addBlock(self, position, id):
        global addedBlocks
        _position = [int(position.x), int(position.y), int(position.z)]
        if _position[0] > CHUNK_WIDTH:
            _position[0] -= CHUNK_WIDTH
            Client.send_message("addBlock", [_position, id, (self.position[0] + 1, self.position[2])])
        elif _position[0] < 1:
            _position[0] += CHUNK_WIDTH
            Client.send_message("addBlock", [_position, id, (self.position[0] - 1, self.position[2])])
        if _position[2] > CHUNK_WIDTH:
            _position[2] -= CHUNK_WIDTH
            Client.send_message("addBlock", [_position, id, (self.position[0], self.position[2]+1)])
        elif _position[2] < 1:
            _position[2] += CHUNK_WIDTH
            Client.send_message("addBlock", [_position, id, (self.position[0], self.position[2]-1)])
        if 0 < _position[0] <= CHUNK_WIDTH and 0 < _position[2] <= CHUNK_WIDTH:
            Client.send_message("addBlock", [_position, id, (self.position[0], self.position[2])])
        #self.checkSurrounding(_position)
        #self.render()

    def checkSurrounding(self, pos):
        chList = []
        for i in itertools.product(range(pos[0] - 1, pos[0] + 2), range(pos[1] - 1, pos[1] + 2),
                                   range(pos[2] - 1, pos[2] + 2)):
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
        for ch in chList:
            ch.render()

    def setCollider(self):
        self.collider = MeshCollider(self, mesh=self.model, center=Vec3(0, 0, 0))
        self.hasCollider = True

    def updateBorder(self):
        global renderedChunkPos
        if (self.position[0] + 1, self.position[2]) in renderedChunkPos:
            self.blockIDs[CHUNK_WIDTH + 1, :, :] = getChunk((self.position[0] + 1, self.position[2])).blockIDs[1, :, :]
        if (self.position[0] - 1, self.position[2]) in renderedChunkPos:
            self.blockIDs[0, :, :] = getChunk((self.position[0] - 1, self.position[2])).blockIDs[CHUNK_WIDTH, :, :]
        if (self.position[0], self.position[2] + 1) in renderedChunkPos:
            self.blockIDs[:, :, CHUNK_WIDTH + 1] = getChunk((self.position[0], self.position[2] + 1)).blockIDs[:, :, 1]
        if (self.position[0], self.position[2] - 1) in renderedChunkPos:
            self.blockIDs[:, :, 0] = getChunk((self.position[0], self.position[2] - 1)).blockIDs[:, :, CHUNK_WIDTH]
        self.getRenderable()
