from ursina import *
import numpy as np
import itertools
from noise import *
from ursina.shaders import *

CHUNK_WIDTH = 8
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

    def getRenderable(self, maxHeight=(CHUNK_HEIGHT - 1)):
        # Get a list of renderable blocks
        mask = (self.blockIDs == 0)

        right = np.argwhere(mask[1:] & (self.blockIDs[:-1] != 0))
        left = np.argwhere(mask[:-1] & (self.blockIDs[1:] != 0)) + np.array([1,0,0])
        top = np.argwhere(mask[:, 1:] & (self.blockIDs[:, :-1] != 0))
        bottom = np.argwhere(mask[:, :-1] & (self.blockIDs[:, 1:] != 0)) + np.array([0,1,0])
        front = np.argwhere(mask[:, :, 1:] & (self.blockIDs[:, :, :-1] != 0))
        back = np.argwhere(mask[:, :, :-1] & (self.blockIDs[:, :, 1:] != 0)) + np.array([0,0,1])
        
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
        global addedBlocks, deletedBlocks
        gradient = [(i * 4) / CHUNK_HEIGHT for i in range(int(CHUNK_HEIGHT / 2), -int(CHUNK_HEIGHT / 2), -1)]
        self.blockIDs = np.zeros((CHUNK_WIDTH + 2, CHUNK_HEIGHT, CHUNK_WIDTH + 2), dtype='int16')

        @np.vectorize
        def caveNoiseGen(x, y, z):
            return 1 if (snoise3(x / 64, y / 64, z / 64, octaves=6) + (1.5 - (gradient[y]))) * 10 > 0 else 0

        @np.vectorize
        def heightNoiseGen(x, y, z):
            return 1 if (snoise3(x / 128 + snoise3(x / 8192, y / 8192, z / 8192, octaves=1), y / 8192,
                                 z / 128, octaves=6) +
                         gradient[y]) * 10 > 0 else 0

        maxHeight = 0
        time1 = time.perf_counter()
        self.blockIDs = np.zeros((CHUNK_WIDTH + 2, CHUNK_HEIGHT, CHUNK_WIDTH + 2), dtype='int16')
        '''x, y, z = np.meshgrid(np.arange(CHUNK_WIDTH + 2) + (self.position[0] * CHUNK_WIDTH) - 1,
                              np.arange(CHUNK_HEIGHT),
                              np.arange(CHUNK_WIDTH + 2) + (self.position[2] * CHUNK_WIDTH) - 1)
        caveNoise = caveNoiseGen(x, y, z)
        heightNoise = heightNoiseGen(x, y, z)
        print(caveNoise[5, 5, 5])
        for i in itertools.product(range(CHUNK_WIDTH + 2), range(CHUNK_WIDTH + 2)):
            xpos = (i[0] + (self.position[0] * CHUNK_WIDTH) - 1)
            zpos = (i[1] + (self.position[2] * CHUNK_WIDTH) - 1)
            for y in range(CHUNK_HEIGHT):
                # heightNoise = (snoise3(xpos / 128 + snoise3(xpos / 8192, y / 8192, zpos / 8192, octaves=1), y / 8192,
                #                       zpos / 128, octaves=6) +
                #               gradient[y]) * 10
                # heightNoise = 1 if heightNoise > 0 else 0
                # caveNoise = (snoise3(xpos / 64, y / 64, zpos / 64, octaves=6) + (1.5 - (gradient[y]))) * 10
                # caveNoise = 1 if caveNoise > 0 else 0
                self.blockIDs[i[0], y, i[1]] = int(heightNoise[y, i[0], i[1]] if caveNoise[y, i[0], i[1]] == 1 else 0)
            self.blockIDs[i[0], 0, i[1]] = 1'''
        posList = np.array([((i[0] + (self.position[0]*CHUNK_WIDTH)-1)/100, (i[1] + (self.position[2]*CHUNK_WIDTH)-1)/100)
                for i in itertools.product(range(CHUNK_WIDTH+2), range(CHUNK_WIDTH+2))])
        noiseVals = np.array([snoise2(x=i[0] + snoise2(x=i[0], y=i[1], octaves=3), y=i[1], octaves=3, base=seeds[0]) for i in posList])
        blockHeights = np.array(np.floor(noiseVals * 15) + 48, dtype='int16')
        #maxHeight = np.max(blockHeights)
        for idx, i in enumerate(itertools.product(range(CHUNK_WIDTH+2), range(CHUNK_WIDTH+2))):
            for y in range(blockHeights[idx]):
                if y <= blockHeights[idx] - 3:
                    self.blockIDs[i[0], y, i[1]] = 1
                else:
                    self.blockIDs[i[0], y, i[1]] = 2
        # Generate added/removed blocks
        if (self.position[0], self.position[2]) in addedBlocks:
            for added in addedBlocks[(self.position[0], self.position[2])]:
                self.blockIDs[added[0], added[1], added[2]] = added[3]
        if (self.position[0], self.position[2]) in deletedBlocks:
            for deleted in deletedBlocks[(self.position[0], self.position[2])]:
                self.blockIDs[deleted[0], deleted[1], deleted[2]] = 0
        # get blocks we need to actually render
        #self.getRenderable()
        print("generate: " + str(time.perf_counter() - time1))
        self.isGenerated = True

    def render(self):
        time1 = time.perf_counter()
        right, left, top, bottom, front, back = self.getRenderable()
        self.unrender()
        self.updateBorder()
        #for block in self.renderBlocks:
        #    self.addToMesh((block + self.position * (CHUNK_WIDTH - 1)),
        #                   int(self.blockIDs[block[0], block[1], block[2]]))
        self.buildMesh(left, right, top, bottom, front, back)
        print(len(self.verts))
        if self.verts is None:
            self.isRendered = True
            return
        self.model = Mesh(vertices=[tuple(i) for i in self.verts], normals=self.norms, uvs=self.uvs.tolist())
        self.texture = blockTex
        self.collider = MeshCollider(self, mesh=self.model, center=Vec3(0, 0, 0))
        self.visible_self = True
        self.isRendered = True
        self.shader = lit_with_shadows_shader
        print("render: " + str(time.perf_counter() - time1))

    def unrender(self):
        self.model = None
        self.verts = None
        self.uvs = None
        self.norms = list()

    def addToMesh(self, pos, blockID=1):
        face_uvs = np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
        _pos = np.array(pos)
        if self.verts is None:
            self.verts = base_verts + _pos
        else:
            self.verts = np.append(self.verts, base_verts + _pos, axis=0)
        for i in range(6):
            if self.uvs is None:
                self.uvs = (face_uvs + texFaceOffsets[blockID, i]) / TEXIMGWIDTH
            else:
                self.uvs = np.append(self.uvs, (face_uvs + texFaceOffsets[blockID, i]) / TEXIMGWIDTH, axis=0)
        self.norms += base_norms

    def buildMesh(self, left, right, top, bottom, front, back):
        face_uvs = np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
        self.verts = np.empty((0,3))
        self.uvs = np.empty((0,2))
        self.norms = []
        offset = self.position * (CHUNK_WIDTH - 1)
        print(left)
        for l in left:
            self.verts = np.append(self.verts, left_face + l + offset, axis=0)
            self.uvs = np.append(self.uvs, (face_uvs + texFaceOffsets[self.blockIDs[l[0], l[1], l[2]], 0]) / TEXIMGWIDTH, axis=0)
            self.norms.append([-1.0, 0.0, 0.0])
        for r in right:
            self.verts = np.append(self.verts, right_face + r + offset, axis=0)
            self.uvs = np.append(self.uvs, (face_uvs + texFaceOffsets[self.blockIDs[r[0], r[1], r[2]], 1]) / TEXIMGWIDTH, axis=0)
            self.norms.append([1.0, 0.0, 0.0])
        for f in front:
            self.verts = np.append(self.verts, front_face + f + offset, axis=0)
            self.uvs = np.append(self.uvs, (face_uvs + texFaceOffsets[self.blockIDs[f[0], f[1], f[2]], 2]) / TEXIMGWIDTH, axis=0)
            self.norms.append([0.0, 0.0, 1.0])
        for b in back:
            self.verts = np.append(self.verts, back_face + b + offset, axis=0)
            self.uvs = np.append(self.uvs, (face_uvs + texFaceOffsets[self.blockIDs[b[0], b[1], b[2]], 3]) / TEXIMGWIDTH, axis=0)
            self.norms.append([0.0, 0.0, -1.0])
        for b in bottom:
            self.verts = np.append(self.verts, bottom_face + b + offset, axis=0)
            self.uvs = np.append(self.uvs, (face_uvs + texFaceOffsets[self.blockIDs[b[0], b[1], b[2]], 4]) / TEXIMGWIDTH, axis=0)
            self.norms.append([0.0, -1.0, 0.0])
        for t in top:
            self.verts = np.append(self.verts, top_face + t + offset, axis=0)
            self.uvs = np.append(self.uvs, (face_uvs + texFaceOffsets[self.blockIDs[t[0], t[1], t[2]], 5]) / TEXIMGWIDTH, axis=0)
            self.norms.append([0.0, 1.0, 0.0])

    def deleteBlock(self, position):
        global deletedBlocks
        _position = [int(position.x), int(position.y), int(position.z)]
        if self.blockIDs[_position[0], _position[1], _position[2]] != 0:
            self.blockIDs[_position[0], _position[1], _position[2]] = 0
        self.renderBlocks = self.renderBlocks[np.all(self.renderBlocks != _position, axis=1)]
        if (self.position[0], self.position[2]) in deletedBlocks:
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
            getChunk((self.position[0] + 1, self.position[2])).addBlock(
                Vec3(_position[0], _position[1], _position[2], ) - Vec3(CHUNK_WIDTH, 0, 0), id)
        if _position[0] < 1:
            getChunk((self.position[0] - 1, self.position[2])).addBlock(
                Vec3(_position[0], _position[1], _position[2], ) + Vec3(CHUNK_WIDTH, 0, 0), id)
        if _position[2] > CHUNK_WIDTH:
            getChunk((self.position[0], self.position[2] + 1)).addBlock(
                Vec3(_position[0], _position[1], _position[2], ) - Vec3(0, 0, CHUNK_WIDTH), id)
        if _position[2] < 1:
            getChunk((self.position[0], self.position[2] - 1)).addBlock(
                Vec3(_position[0], _position[1], _position[2], ) + Vec3(0, 0, CHUNK_WIDTH), id)
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
            # if not arreq_in_list(np.array(newpos), ch.renderBlocks):
            #    ch.checkRenderable(tuple(newpos))
        for ch in chList:
            ch.getRenderable()
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
