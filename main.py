from Player import *

from Inventory import *

app = Ursina()

from Chunk import *

load_model('block')

options = {"renderDistance": 4, "shadows": False, }

fpc = Player(x=0.5, y=256, z=0.5, height=0.85, jump_duration=0.2, jump_height=1.2, gravity=0.8)
# collision_zone = CollisionZone(parent=fpc, radius=16)
blockHighlight = Entity(model='wireframe_cube', thickness=3, visible=False, position=(0, 0, 0), scale=1.1,
                        color=color.rgba(64, 64, 64), origin=(0.45, 0.45, 0.45), unlit=True)
coordText = Text()
inventory = Inventory(enabled=False)
sessionID = None
p = None
otherPlayerEntities = dict()


def recvChunkBlocks(Content):
    print("received chunk")
    pos = Content[0]
    if pos in renderedChunkPos:
        ch = getChunk(pos)
        ch.blockIDs = Content[1]
        ch.isGenerated = True
        ch.render()


def recvSessionId(Content):
    print("Recieved session ID")
    global sessionID
    sessionID = Content


def preGenProgress(Content):
    prog = Content[0]
    total = Content[1]
    print("Server is pregenerating: ", prog, "/", total)


def serverPrint(Content):
    print(Content)


def onConnectionError(Reason):
    print(f"Error ! Reason : {Reason}")


def posUpdate(Content):
    playerID = Content[0]
    position = Content[1]
    if not playerID == sessionID:
        if abs(position[0] - fpc.position.x) < options["renderDistance"] * CHUNK_WIDTH and abs(
                position[2] - fpc.position.y) < options["renderDistance"] * CHUNK_WIDTH:
            if not playerID in otherPlayerEntities:
                otherPlayerEntities[playerID] = Entity(model='cube', color=color.red)
            otherPlayerEntities[playerID].position = position


def playerJoined():
    print("A player joined!")


def allPositions(Content):
    print("All positions recieved")
    for k in Content.keys():
        if not k == sessionID:
            if abs(Content[k][0] - fpc.position.x) < options["renderDistance"] * CHUNK_WIDTH and abs(
                    Content[k][2] - fpc.position.z) < options["renderDistance"] * CHUNK_WIDTH:
                if not k in otherPlayerEntities:
                    otherPlayerEntities[k] = Entity(model='cube', color=color.red)
                otherPlayerEntities[k].position = Content[k]


dl = DirectionalLight(y=2, z=3, shadows=True, rotation_x=45, rotation_y=45, rotation_z=45, enabled=False)
al = AmbientLight(enabled=True)

def doChunkRendering(_currentChunk):
    xRange = range(_currentChunk[0] - options["renderDistance"], _currentChunk[0] + options["renderDistance"] + 1)
    zRange = range(_currentChunk[1] - options["renderDistance"], _currentChunk[1] + options["renderDistance"] + 1)
    chRange = sorted(list(itertools.product(xRange, zRange)),
                     key=lambda x: abs(x[0] - _currentChunk[0]) + abs(x[1] - _currentChunk[1]))
    for idx, chunk in enumerate(renderedChunkPos):
        if not (chunk[0] in xRange and chunk[1] in zRange):
            destroy(renderedChunks[idx])
            del renderedChunks[idx]
            del renderedChunkPos[idx]
            client.send_message("unloadChunk", chunk)
            return

    if len(renderedChunks) == 0:
        renderedChunkPos.append(chRange[0])
        renderedChunks.append(Chunk(chRange[0], options["shadows"]))
        renderedChunks[-1].generate()
    else:
        for i in chRange:
            if not i in renderedChunkPos:
                if renderedChunks[-1].isGenerated:
                    renderedChunkPos.append((i[0], i[1]))
                    renderedChunks.append(Chunk((i[0], i[1])))
                    renderedChunks[-1].generate()
                    return

    dl.shadows = True


def input(key):
    global mouseChunk, lookingAt, p, isMenu
    if not isMenu:
        if key == "left mouse down" and lookingAt is not None:
            getChunk(mouseChunk).deleteBlock(
                Vec3(lookingAt.x - (CHUNK_WIDTH * mouseChunk[0]), lookingAt.y,
                     lookingAt.z - (CHUNK_WIDTH * mouseChunk[1])))
        elif key == "right mouse down" and lookingAt is not None:
            getChunk(mouseChunk).addBlock(Vec3(lookingAt.x - ((CHUNK_WIDTH) * mouseChunk[0]) + mouse.normal.x,
                                               lookingAt.y + mouse.normal.y,
                                               lookingAt.z - ((CHUNK_WIDTH) * mouseChunk[1]) + mouse.normal.z), 1)
        if key == 'escape':
            ingameOptionsMenu()
            mouse.locked = False
            fpc.mouse_sensitivity = Vec2(0, 0)
            isMenu = True
            '''client.client.client.close()
            if p is not None:
                p.terminate()
            exit(0)'''

        if key == 'q':
            mouse.locked = False

        if key == 'e':
            inventory.enabled = not inventory.enabled
            mouse.locked = not inventory.enabled
            fpc.mouse_sensitivity = Vec2(0, 0) if inventory.enabled else Vec2(40, 40)
    else:
        if key == 'escape':
            mouse.locked = True
            renderDistanceSlider.disable()
            toggleShadowsButton.disable()
            quitButton.disable()
            fpc.mouse_sensitivity = Vec2(40, 40)
            isMenu = False

last_position = [0, 0, 0]
isMenu = True
lastNetTime = 0


def update():
    global currentChunk, lookingAt, mouseChunk, chunkThread, last_position, lastNetTime
    if not isMenu:
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

        '''if mouse.left and lookingAt is not None:
            print("lmb")
            getChunk(mouseChunk).deleteBlock(
                Vec3(lookingAt.x - (CHUNK_WIDTH * mouseChunk[0]), lookingAt.y,
                     lookingAt.z - (CHUNK_WIDTH * mouseChunk[1])))
        elif mouse.right and lookingAt is not None:
            getChunk(mouseChunk).addBlock(Vec3(lookingAt.x - ((CHUNK_WIDTH) * mouseChunk[0]) + mouse.normal.x,
                                               lookingAt.y + mouse.normal.y,
                                               lookingAt.z - ((CHUNK_WIDTH) * mouseChunk[1]) + mouse.normal.z), 1)'''

        if held_keys["control"]:
            fpc.speed = 10
            camera.fov = 110
        else:
            fpc.speed = 5
            camera.fov = 90

        '''if held_keys["space"]:
            fpc.y += 5 * time.dt
        if held_keys["shift"]:
            fpc.y -= 5 * time.dt'''

        dl.x = currentChunk[0] * CHUNK_WIDTH
        dl.z = currentChunk[1] * CHUNK_WIDTH
        client.process_net_events()

        if last_position != [fpc.position[0], fpc.position[1], fpc.position[2]]:
            last_position = [fpc.position[0], fpc.position[1], fpc.position[2]]
            client.send_message("posUpdate", [sessionID, [fpc.position[0], fpc.position[1], fpc.position[2]]])

        coordText.text = str(int(fpc.position.x)) + ", " + str(int(fpc.position.y)) + ", " + str(int(fpc.position.z))

sky = Sky(color="87ceeb", texture=None)

ipAddress = "localhost"
loadingText = Text("Starting server", enabled=False)
newWorldButton = Button("Create World", scale=(0.5, 0.1), y=0.1)
worldNameText = TextField(text='New World')
joinServerButton = Button("Join Server", scale=(0.5, 0.1), y=0.1)
serverIPText = TextField(text='localhost')
loadWorldButtons = None

def setRenderDistance():
    options["renderDistance"] = int(renderDistanceSlider.value)

def setShadows():
    print(options["shadows"])
    options["shadows"] = not options["shadows"]
    toggleShadowsButton.text = "Shadows: " + str(options["shadows"])
    if options["shadows"]:
        dl.enable()
        al.disable()
        for ch in renderedChunks:
            ch.shader = lit_with_shadows_shader
    else:
        dl.disable()
        al.enable()
        for ch in renderedChunks:
            ch.shader = None

def quitGame():
    global renderedChunks, renderedChunkPos, isMenu
    client.send_message("playerQuit", sessionID)
    if p is not None:
        client.send_message("stop", "")
    for ch in renderedChunks:
        destroy(ch)
    renderedChunks = []
    renderedChunkPos = []
    isMenu = True
    coordText.disable()
    quitButton.disable()
    showMenu()

renderDistanceSlider = ThinSlider(min=2, max=16, step=1, text="Render Distance", enabled=False, on_value_changed=setRenderDistance, y=0.1)
toggleShadowsButton = Button(text="Shadows: " + str(options["shadows"]), scale=(0.5,0.1), y=0, on_click=setShadows, enabled=False)
quitButton = Button(text="Save and Quit", scale=(0.5,0.1), y=-0.2, on_click=quitGame, enabled=False)

def addEvents():
    client.client.event(recvChunkBlocks)
    client.client.event(recvSessionId)
    client.client.event(onConnectionError)
    client.client.event(allPositions)
    client.client.event(playerJoined)
    client.client.event(posUpdate)
    client.client.event(serverPrint)


def startGame(worldName):
    global p, isMenu
    print(worldName)
    addEvents()
    newWorldButton.disable()
    worldNameText.disable()
    optionsBackButton.disable()
    loadingText.enable()
    loadingText.text = "Starting server"
    print("Starting server")
    print([sys.executable, 'server.py', worldName])
    p = subprocess.Popen([sys.executable, 'server.py', worldName], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # Start the server
    print("Started server")
    isMenu = False
    print("Connecting...")
    loadingText.text = "Connecting..."
    # Wait for connection
    while not client.connected():
        pass
    print("Connected")
    print("Loading world...")
    loadingText.text = "Loading world..."
    while len(renderedChunks) < 2:
        doChunkRendering((0, 0))
        client.process_net_events()
    destroy(loadingText)
    fpc.enable()
    fpc.y = max(np.argwhere(getChunk((0, 0)).blockIDs[0, :, 0] != 0)) + 30
    # app.run()


def loadWorld(worldName):
    if loadWorldButtons is not None:
        loadWorldButtons.disable()
    print("Starting client")
    client.connect('localhost')
    startGame(worldName)


def newWorld():
    worldName = worldNameText.text
    worldNameText.disable()
    newWorldButton.disable()
    print("Starting client")
    client.connect('localhost')
    startGame(worldName)


def joinServer():
    global ipAddress, isMenu
    ipAddress = serverIPText.text
    serverIPText.disable()
    joinServerButton.disable()
    client.connect(ipAddress)
    addEvents()
    isMenu = False
    print("Connecting...")
    loadingText.text = "Connecting..."
    # Wait for connection
    while not client.connected():
        pass
    print("Connected")
    print("Loading world...")
    loadingText.text = "Loading world..."
    while len(renderedChunks) < 2:
        doChunkRendering((0, 0))
        client.process_net_events()
    destroy(loadingText)
    fpc.enable()


def newWorldMenu():
    global isMenu
    isMenu = True
    mainMenu.disable()
    newWorldButton.on_click = newWorld
    newWorldButton.enable()
    worldNameText.enable()
    optionsBackButton.enable()


def loadWorldMenu():
    global isMenu, loadWorldButtons
    isMenu = True
    mainMenu.disable()
    worldFolders = os.listdir("worlds")
    worldButtonList = dict()
    for f in worldFolders:
        worldButtonList[f] = Func(loadWorld, f)
    print(worldButtonList)
    loadWorldButtons = ButtonList(worldButtonList)
    optionsBackButton.enable()


def optionsMenu():
    global isMenu
    isMenu = True
    renderDistanceSlider.enable()
    toggleShadowsButton.enable()
    optionsBackButton.enable()
    mainMenu.disable()

def ingameOptionsMenu():
    global isMenu
    isMenu = True
    renderDistanceSlider.enable()
    toggleShadowsButton.enable()
    quitButton.enable()

def joinServerMenu():
    global isMenu
    isMenu = True
    mainMenu.disable()
    serverIPText.enable()
    joinServerButton.enable()
    joinServerButton.on_click = joinServer
    optionsBackButton.enable()


mainMenu = ButtonList({'New World': newWorldMenu, 'Load World': loadWorldMenu, 'Join Server': joinServerMenu, 'Options': optionsMenu},
                      enabled=False)

def showMenu():
    isMenu = True
    fpc.disable()
    # A play button that show the loading menu when clicked
    newWorldButton.disable()
    worldNameText.disable()
    joinServerButton.disable()
    serverIPText.disable()
    renderDistanceSlider.disable()
    toggleShadowsButton.disable()
    optionsBackButton.disable()
    if loadWorldButtons is not None:
        loadWorldButtons.disable()
    mainMenu.enable()

optionsBackButton = Button(text="Back", on_click=showMenu, scale=0.1, y=-0.1)

if __name__ == '__main__':
    window.fps_counter.enabled = False
    window.vsync = False
    isMenu = True
    showMenu()
    app.run()
