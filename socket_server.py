
import sys
import socket
import threading
import cv2
import numpy as np
import time

import os
# import toothNet2
from autoplanning import toothNet2

# from interface import
from network.interface import InterfaceHeader, AfmMat, ToothDiagnosis, PyToothDiag, Packet
from common.controller import AiContoller, get_ai_controller, write_buffer
connections = []
total_connections = 0
# https://github.com/pricheal/python-client-server

IMAGE_MSG_PROTOCOL = "image"
BBOX_MSG_PROTOCOL = "bbox"


HEADER_SIZE = InterfaceHeader.get_byte_size()


class Receiver(object):
    def __init__(self):
        self.concat_bytes = bytearray()
        self.logger = toothNet2.get_runtime_logger()
        self.total_received_size = 0
        self.total_data_size = 0 # header + body
        self._finished = True
        self._writed_header = False

        self.header = InterfaceHeader()

        self.bodybuffer = None

        self.header_list = []
        self.bytes_list = []

    def is_finished(self):
        return self._finished

    def pop(self):
        if self.header_list and self.bytes_list:
            return self.header_list.pop(0), self.bytes_list.pop(0)

    def receive_packet(self, packet_bytes:bytearray):
        """
        :param packet_bytes: packate bytes
        :return: remain bytes size
        """

        # new session begin
        if self._finished:
            self._finished = False
            self._writed_header = False

        receive_size = len(packet_bytes)
        self.total_received_size += receive_size
        self.concat_bytes += packet_bytes

        self.logger.debug("{}/{} : receive{}".format(self.total_received_size, self.total_data_size, receive_size))

        if self.total_received_size >= HEADER_SIZE:
            if not self._writed_header:
                self._writed_header = True

                decode_str = bytes(self.concat_bytes[:HEADER_SIZE])
                self.header.decoding(decode_str, len(decode_str))

                header = InterfaceHeader()
                header.decoding(decode_str, len(decode_str))
                self.header_list.append(header)

                bodysize = self.header.getBodySize()
                index = self.header.getIndex()
                msg = self.header.getMessage()

                self.logger.info("check taken msg : {} - {:03d} body size : {}".format(msg, index, bodysize))

                self.total_data_size = HEADER_SIZE + bodysize
                # if bodysize == 0:
                #     self._finished = True
                #     self.total_received_size = 0
                # else:
                #     pass

            remain_size = self.total_data_size - self.total_received_size
        else:
            remain_size = HEADER_SIZE - self.total_received_size

        if self.total_received_size == self.total_data_size:
            self.bytes_list.append(self.concat_bytes[HEADER_SIZE:])
            self.total_received_size = 0
            self._finished = True
            self._writed_header = False
            self.concat_bytes = bytearray()

        return remain_size



# Client class, new instance created for each connected client
# Each instance has the socket and address that is associated with items
# Along with an assigned ID and a name chosen by the client
class Client(threading.Thread):
    def __init__(self, socket, address, id, name, signal):

        threading.Thread.__init__(self)
        self.socket = socket
        self.address = address
        self.id = id
        self.name = name
        self.signal = signal
        self.logger = toothNet2.get_runtime_logger()


    def __str__(self):
        return str(self.id) + " " + str(self.address)

    # Attempt to get data from client
    # If unable to, assume client has disconnected and remove him from server data
    # If able to and we get data back, print it in the server and send it back to every
    # client aside from the client that has sent it
    # .decode is used to convert the byte data into a printable string
    def run(self):

        concat_data = bytearray()
        to_taken_filesize = 0
        remain_size = InterfaceHeader.get_byte_size()
        header_size = InterfaceHeader.get_byte_size()
        index = 0
        strlen = 0



        receiver = Receiver()

        controller = get_ai_controller()

        while self.signal:
            try:
                remain_size = max(remain_size, 1)
                data = self.socket.recv(remain_size)
                # concat_data += data
            except:
                self.logger.info("Client " + str(self.address) + " has disconnected")
                self.signal = False
                connections.remove(self)
                break
            if data != "":


                remain_size = receiver.receive_packet(data)
                # receive complete
                if receiver.is_finished():
                    header, body_buffer = receiver.pop()
                    self.logger.debug("receive complete")

                    post_message = controller.process(header, body_buffer)

                    if isinstance(post_message, Packet):
                        post_message.get_packet_bytes()

                    concat_bytes = post_message.get_packet_bytes()

                    try:
                        self.logger.info("sending packet")
                        self.socket.send(concat_bytes)

                    except ConnectionResetError as e:
                        self.logger.critical("ConnectionResetError {} : {}".format(e.errno, e.strerror))


    def image_process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)
        # cv2.imshow("", edges)
        return edges

    def image_encoding_write(self, image):
        rszimg = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10))
        encode_buffer = cv2.imencode(".png", rszimg)
        result, img_ability_png = cv2.imencode('.png', rszimg)
        self.logger.debug("encoded image:", encode_buffer)

        rsz_encode = cv2.imdecode(img_ability_png, cv2.IMREAD_UNCHANGED)
        cv2.imwrite("rszimag.png", rsz_encode)


def newConnections(socket):
    while True:
        sock, address = socket.accept()
        global total_connections
        connections.append(Client(sock, address, total_connections, "Name", True))
        connections[len(connections) - 1].start()
        logger = toothNet2.get_runtime_logger()
        logger.info("New connection at ID " + str(connections[len(connections) - 1]))
        total_connections += 1


def socekt_server_run():
    # Get host and port
    # host = input("Host: ")
    # port = int(input("Port: "))
    # driving ai-core
    con = get_ai_controller()

    logger = toothNet2.get_runtime_logger()
    host = ""
    port = 5566

    # Create new server socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5)
    logger.info(sock.getsockname())
    logger.info(socket.gethostname())
    logger.info(socket.gethostbyname(socket.gethostname()))

    logger.info(socket.AF_INET)
    # Create new thread to wait for connections
    newConnectionsThread = threading.Thread(target=newConnections, args=(sock,))
    newConnectionsThread.start()



if __name__ == "__main__":
    socekt_server_run()