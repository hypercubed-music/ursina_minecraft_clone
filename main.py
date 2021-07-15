import threading
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

RENDER_DISTANCE = 4
BLOCK_TYPES = 2

app = Ursina()

from Chunk import *
coords = Text(text="", origin=(0.75,0.75), background=True)

load_model('block')

fpc = FirstPersonController(x=0, y=256, z=0, height=1.7, jump_duration=0.2, jump_height=1.2)
blockHighlight = Entity(model='wireframe_cube', thickness=3, visible=False, position=(0,0,0), scale=1.1, color=color.rgba(64, 64, 64), origin=(0.45, 0.45, 0.45))

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

@Client.event
def recvChunkBlocks(Content):
    pos = (Content[0], Content[1])
    print("Recieved a chunk ", pos)
    if pos in renderedChunkPos:
        getChunk(pos).blockIDs = Content[2]
        getChunk(pos).isGenerated = True

@Client.event
def serverPrint(Content):
    print(Content)

dl = DirectionalLight(y=2, z=3, shadows=True, rotation_x = 45, shadow_map_resolution = (4096,4096))
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
        print("Render")
        #threading.Thread(renderedChunks[-1].render(), daemon=True).start()
        renderedChunks[-1].render()
    else:
        for i in itertools.product(xRange, zRange):
            if not (i[0], i[1]) in renderedChunkPos:
                if len(renderedChunks) == 0:
                    renderedChunkPos.append((i[0], i[1]))
                    renderedChunks.append(Chunk((i[0], i[1])))
                    # threading.Thread(renderedChunks[-1].generate(), daemon=True).start()
                    renderedChunks[-1].generate()
                else:
                     if renderedChunks[-1].isGenerated:
                        renderedChunkPos.append((i[0], i[1]))
                        renderedChunks.append(Chunk((i[0], i[1])))
                        #threading.Thread(renderedChunks[-1].generate(), daemon=True).start()
                        renderedChunks[-1].generate()
                        break

    if not renderedChunks[-1].hasCollider:
        renderedChunks[-1].setCollider()
        # recalculate shadows
        dl.shadows = True

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
    '''if chunkThread is None:
        chunkThread = threading.Thread(doChunkRendering(currentChunk), daemon=True).start()
    elif not chunkThread.is_alive():
        chunkThread = threading.Thread(doChunkRendering(currentChunk), daemon=True).start()'''
    doChunkRendering(currentChunk)
    if mouse.hovered_entity is not None and 1 < distance(mouse.point, fpc) < 10:
        blockHighlight.visible = True
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
    dl.rotation_x -= 0.1 * time.dt
    Client.process_net_events()

# Start the server
p = subprocess.Popen([sys.executable, 'generate_proc.py'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
print("Client has started the server")

sky = Sky(color="87ceeb", texture=None)
# Wait for connection
while not Client.connected:
    pass
while len(renderedChunks) < math.pow(RENDER_DISTANCE * 2 + 1, 2):
    doChunkRendering((0,0))
    Client.process_net_events()
    #print(len(renderedChunks), "/", math.pow(RENDER_DISTANCE * 2 + 1, 2))
    '''output = p.stdout.readline()
    if output == '' and p.poll() is not None:
        break
    if output:
        print(output.strip())'''
#fpc = EditorCamera(enabled=1)
fpc.y = max(np.argwhere(getChunk((0, 0)).blockIDs[1, :, 1] != 0)) + 10

app.run()