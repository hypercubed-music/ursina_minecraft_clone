from ursinanetworking import *

class GameClient:
    def __init__(self, ip='localhost', port=25565):
        self.ip = ip
        self.port = port
        self.client = UrsinaNetworkingClient(ip, port)

    def connect(self, ip):
        self.client = UrsinaNetworkingClient(ip, self.port)

    def changePort(self, port):
        self.client = UrsinaNetworkingClient(self.ip, port)

    def connected(self):
        return self.client.connected

    def send_message(self, message, content):
        self.client.send_message(message, content)

    def process_net_events(self):
        self.client.process_net_events()