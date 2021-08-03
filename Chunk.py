import time

from ursina import *
import numpy as np
import itertools
from ursina.shaders import *
from client import GameClient

CHUNK_WIDTH = 16
CHUNK_HEIGHT = 256
TEXIMGHEIGHT = 16
TEXIMGWIDTH = 32

blockTex = load_texture('assets/atlas.png')
# texOffsets = [None, (0,1), (0,0), (1,0), (1,1)]
texFaceOffsets = np.array([[(0, 31), (0, 31), (0, 31), (0, 31), (0, 31), (0, 31)],  # air?
                           [(19, 31) for x in range(6)],  # stone
                           [(11, 30) for x in range(6)], # dirt
                           [(3, 31), (3, 31), (3, 31), (3, 31), (11, 30), (2, 31)],  # grass
                           [(28, 29), (28, 29), (28, 29), (28, 29), (29, 29), (29, 29)],  # wood
                           [(22, 27) for x in range(6)],  # leaves
                           [(23, 22) for x in range(6)], # wool
                           [(23, 22) for x in range(6)],
                           [(24, 22) for x in range(6)],
                           [(25, 22) for x in range(6)],
                           [(26, 22) for x in range(6)],
                           [(27, 22) for x in range(6)],
                           [(28, 22) for x in range(6)],
                           [(29, 22) for x in range(6)],
                           [(30, 22) for x in range(6)],
                           [(31, 22) for x in range(6)],
                           [(1, 21) for x in range(6)],
                           [(2, 21) for x in range(6)],
                           [(3, 21) for x in range(6)],
                           [(4, 21) for x in range(6)],
                           [(5, 21) for x in range(6)],
                           [(6, 21) for x in range(6)],
                           ])
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

#Client = UrsinaNetworkingClient("localhost", 25565)
client = GameClient("localhost", 25565)

def getChunk(pos):
    if pos in renderedChunkPos:
        return renderedChunks[renderedChunkPos.index(pos)]

class Chunk(Entity):
    def __init__(self, position=(0, 0), seed=0):
        super().__init__(visible_self=False, position=(position[0] * CHUNK_WIDTH, 0, position[1] * CHUNK_WIDTH))
        self.renderBlocks = list()
        # self.renderFaces = list()
        self.blockIDs = np.zeros((CHUNK_WIDTH + 2, CHUNK_HEIGHT, CHUNK_WIDTH + 2), dtype='uint8')
        self.position = np.array([position[0], 0, position[1]])
        self.isGenerated = False
        self.isRendered = False
        self.hasCollider = False
        self.verts = None
        self.uvs = None
        self.norms = list()
        #self.color = color.rgb(random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        self.shader = lit_with_shadows_shader

    def getRenderable(self):
        # Get a list of renderable blocks
        mask = (self.blockIDs == 0)
        right = np.argwhere(mask[1:] & (self.blockIDs[:-1] != 0))
        right = right[(right[:,0] > 0) & (right[:,0] <= CHUNK_WIDTH) & (right[:,2] > 0) & (right[:,2] <= CHUNK_WIDTH)]
        left = np.argwhere(mask[:-1] & (self.blockIDs[1:] != 0)) + np.array([1, 0, 0])
        left = left[(left[:, 0] > 0) & (left[:, 0] <= CHUNK_WIDTH) & (left[:, 2] > 0) & (left[:, 2] <= CHUNK_WIDTH)]
        top = np.argwhere(mask[:, 1:] & (self.blockIDs[:, :-1] != 0))
        top = top[(top[:, 0] > 0) & (top[:, 0] <= CHUNK_WIDTH) & (top[:, 2] > 0) & (top[:, 2] <= CHUNK_WIDTH)]
        bottom = np.argwhere(mask[:, :-1] & (self.blockIDs[:, 1:] != 0)) + np.array([0, 1, 0])
        bottom = bottom[(bottom[:, 0] > 0) & (bottom[:, 0] <= CHUNK_WIDTH) & (bottom[:, 2] > 0) & (bottom[:, 2] <= CHUNK_WIDTH)]
        front = np.argwhere(mask[:, :, 1:] & (self.blockIDs[:, :, :-1] != 0))
        front = front[(front[:, 0] > 0) & (front[:, 0] <= CHUNK_WIDTH) & (front[:, 2] > 0) & (front[:, 2] <= CHUNK_WIDTH)]
        back = np.argwhere(mask[:, :, :-1] & (self.blockIDs[:, :, 1:] != 0)) + np.array([0, 0, 1])
        back = back[(back[:, 0] > 0) & (back[:, 0] <= CHUNK_WIDTH) & (back[:, 2] > 0) & (back[:, 2] <= CHUNK_WIDTH)]

        return right, left, top, bottom, front, back

    def generate(self):
        global addedBlocks, deletedBlocks
        if client.connected():
            _pos = (int(self.position.x), int(self.position.y), int(self.position.z))
            client.send_message("generate", [_pos, CHUNK_WIDTH, CHUNK_HEIGHT])

    def render(self):
        time1 = time.perf_counter()
        right, left, top, bottom, front, back = self.getRenderable()
        self.unrender()
        self.updateBorder()
        self.buildMesh(left, right, top, bottom, front, back)
        if self.verts is None:
            self.isRendered = True
            return
        self.model = Mesh(vertices=self.verts,uvs=self.uvs.tolist(), static=True)
        self.texture = blockTex
        self.visible_self = True
        self.isRendered = True
        print("adding collider")
        self.collider = MeshCollider(self, mesh=self.model, center=Vec3(0, 0, 0))
        self.hasCollider = True
        print("render: ", time.perf_counter()-time1)

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
        client.send_message("deleteBlock", [_position, (self.position[0], self.position[2])])
        # update surrounding blocks
        # self.checkSurrounding(_position)
        # self.render()

    def addBlock(self, position, id):
        global addedBlocks
        _position = [int(position.x), int(position.y), int(position.z)]
        if _position[0] > CHUNK_WIDTH:
            _position[0] -= CHUNK_WIDTH
            client.send_message("addBlock", [_position, id, (self.position[0] + 1, self.position[2])])
        elif _position[0] < 1:
            _position[0] += CHUNK_WIDTH
            client.send_message("addBlock", [_position, id, (self.position[0] - 1, self.position[2])])
        if _position[2] > CHUNK_WIDTH:
            _position[2] -= CHUNK_WIDTH
            client.send_message("addBlock", [_position, id, (self.position[0], self.position[2] + 1)])
        elif _position[2] < 1:
            _position[2] += CHUNK_WIDTH
            client.send_message("addBlock", [_position, id, (self.position[0], self.position[2] - 1)])
        if 0 < _position[0] <= CHUNK_WIDTH and 0 < _position[2] <= CHUNK_WIDTH:
            client.send_message("addBlock", [_position, id, (self.position[0], self.position[2])])
        # self.checkSurrounding(_position)
        # self.render()

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
