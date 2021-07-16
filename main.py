import threading
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

RENDER_DISTANCE = 3
BLOCK_TYPES = 2

app = Ursina()

from Chunk import *

coords = Text(text="", origin=(0.75, 0.75), background=True)

load_model('block')

fpc = FirstPersonController(x=0, y=256, z=0, height=1.7, jump_duration=0.2, jump_height=1.2)
collision_zone = CollisionZone(parent=fpc, radius=16)
blockHighlight = Entity(model='wireframe_cube', thickness=3, visible=False, position=(0, 0, 0), scale=1.1,
                        color=color.rgba(64, 64, 64), origin=(0.45, 0.45, 0.45), unlit=True)
sessionID = None

otherPlayerEntities = dict()

@Client.event
def recvChunkBlocks(Content):
    time1 = time.perf_counter()
    pos = Content[0]
    if pos in renderedChunkPos:
        ch = getChunk(pos)
        ch.blockIDs = Content[1]
        ch.isGenerated = True
        threading.Thread(ch.render(), daemon=True).start()

@Client.event
def recvSessionId(Content):
    print("Recieved session ID")
    global sessionID
    sessionID = Content

@Client.event
def preGenProgress(Content):
    prog = Content[0]
    total = Content[1]
    print("Server is pregenerating: ", prog, "/", total)

@Client.event
def serverPrint(Content):
    print(Content)

@Client.event
def onConnectionError(Reason):
    print(f"Error ! Reason : {Reason}")

@Client.event
def posUpdate(Content):
    print("Recieved position update")
    playerID = Content[0]
    position = Content[1]
    if not playerID == sessionID:
        if abs(position[0] - fpc.position.x) < RENDER_DISTANCE * CHUNK_WIDTH and abs(position[2] - fpc.position.y) < RENDER_DISTANCE * CHUNK_WIDTH:
            if not playerID in otherPlayerEntities:
                otherPlayerEntities[playerID] = Entity(model='cube', color=color.red)
            otherPlayerEntities[playerID].position = position

@Client.event
def playerJoined():
    print("A player joined!")

@Client.event
def allPositions(Content):
    print("All positions recieved")
    for k in Content.keys():
        if not k == sessionID:
            if abs(Content[k][0] - fpc.position.x) < RENDER_DISTANCE * CHUNK_WIDTH and abs(Content[k][2] - fpc.position.z) < RENDER_DISTANCE * CHUNK_WIDTH:
                if not k in otherPlayerEntities:
                    otherPlayerEntities[k] = Entity(model='cube', color=color.red)
                otherPlayerEntities[k].position = Content[k]

dl = DirectionalLight(y=2, z=3, shadows=True, rotation_x=45, rotation_y=45, rotation_z=45)

def doChunkRendering(_currentChunk):
    # unload distant chunks
    xRange = range(_currentChunk[0] - RENDER_DISTANCE, _currentChunk[0] + RENDER_DISTANCE + 1)
    zRange = range(_currentChunk[1] - RENDER_DISTANCE, _currentChunk[1] + RENDER_DISTANCE + 1)
    chRange = sorted(list(itertools.product(xRange, zRange)), key=lambda x: abs(x[0]) + abs(x[1]))
    for idx, chunk in enumerate(renderedChunkPos):
        if not (chunk[0] in xRange and chunk[1] in zRange):
            destroy(renderedChunks[idx])
            del renderedChunks[idx]
            del renderedChunkPos[idx]
            break

    if len(renderedChunks) == 0:
        renderedChunkPos.append(chRange[0])
        renderedChunks.append(Chunk(chRange[0]))
        # threading.Thread(renderedChunks[-1].generate(), daemon=True).start()
        renderedChunks[-1].generate()
    else:
        for i in chRange:
            if not i in renderedChunkPos:
                if renderedChunks[-1].isGenerated:
                    renderedChunkPos.append((i[0], i[1]))
                    renderedChunks.append(Chunk((i[0], i[1])))
                    # threading.Thread(renderedChunks[-1].generate(), daemon=True).start()
                    renderedChunks[-1].generate()
                    break

    if not renderedChunks[-1].hasCollider:
        renderedChunks[-1].setCollider()
        # recalculate shadows
        dl.shadows = True


def input(key):
    global mouseChunk, lookingAt
    if key == "left mouse down" and lookingAt is not None:
        getChunk(mouseChunk).deleteBlock(
            Vec3(lookingAt.x - (CHUNK_WIDTH * mouseChunk[0]), lookingAt.y, lookingAt.z - (CHUNK_WIDTH * mouseChunk[1])))
    elif key == "right mouse down" and lookingAt is not None:
        getChunk(mouseChunk).addBlock(Vec3(lookingAt.x - ((CHUNK_WIDTH) * mouseChunk[0]) + mouse.normal.x,
                                           lookingAt.y + mouse.normal.y,
                                           lookingAt.z - ((CHUNK_WIDTH) * mouseChunk[1]) + mouse.normal.z), 1)
    if key == 'escape':
        Client.client.close()
        p.terminate()
        exit(0)

    if key == 'q':
        mouse.locked = False


chunkThread = threading.Thread()
last_position = [0,0,0]

def update():
    global currentChunk, lookingAt, mouseChunk, chunkThread, last_position
    currentChunk = (math.floor(fpc.position[0] / CHUNK_WIDTH), math.floor(fpc.position[2] / CHUNK_WIDTH))
    doChunkRendering(currentChunk)
    if mouse.hovered_entity is not None and 1 < distance(mouse.point, fpc) < 10:
        blockHighlight.visible = True
        lookingAt = Vec3(math.floor(mouse.world_point.x - 0.5 * mouse.normal.x),
                         math.floor(mouse.world_point.y - 0.5 * mouse.normal.y),
                         math.floor(mouse.world_point.z - 0.5 * mouse.normal.z))
        blockHighlight.position = lookingAt + Vec3(1, 1, 1)
        mouseChunk = (math.floor((lookingAt[0] - 1) / CHUNK_WIDTH), math.floor((lookingAt[2] - 1) / CHUNK_WIDTH))
    else:
        blockHighlight.visible = False
        lookingAt = None

    if held_keys["control"]:
        fpc.speed = 10
        camera.fov = 110
    else:
        fpc.speed = 5
        camera.fov = 90

    coords.text = ", ".join([str(int(i)) for i in list(fpc.position)]) + "\n" + (
        str(list(lookingAt)) if lookingAt is not None else "")
    dl.x = currentChunk[0] * CHUNK_WIDTH
    dl.z = currentChunk[1] * CHUNK_WIDTH
    Client.process_net_events()

    if last_position != [fpc.position[0], fpc.position[1], fpc.position[2]]:
        last_position = [fpc.position[0], fpc.position[1], fpc.position[2]]
        Client.send_message("posUpdate", [sessionID, [fpc.position[0], fpc.position[1], fpc.position[2]]])


# Start the server
'''p = subprocess.Popen([sys.executable, 'server.py'],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)'''
print("Client has started the server")

sky = Sky(color="87ceeb", texture=None)
# Wait for connection
while not Client.connected:
    pass
print("Connected")
while len(renderedChunks) < 2:
    doChunkRendering((0, 0))
    Client.process_net_events()

fpc.y = max(np.argwhere(getChunk((0, 0)).blockIDs[0, :, 0] != 0)) + 20
app.run()
