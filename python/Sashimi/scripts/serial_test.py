"""
A script to choose a Serial port and send messages through it
"""
from time import sleep
import serial
from serial.tools import list_ports


def send_msg(_port, msg):
    """
    sends a message through the port
    """
    _port.write((msg + "\n").encode())


def read(_port):
    """
    read messages from port
    """
    lines = []
    buffer = []
    while _port.in_waiting > 0:
        reading = _port.read()
        for byte in reading:
            if byte != ord('\n') and byte != ord('\r'):
                buffer.append(chr(byte))
            else:
                lines.append(''.join(buffer))
                buffer = []
    return lines


if __name__ == "__main__":
    ports = list(list_ports.comports())
    for n, port in enumerate(ports):
        print(n, port.device)
    n = int(input("device number : "))

    ser = serial.Serial(ports[n].device, 115_200)
    if not ser.isOpen():
        ser.open()
    messages = [
        "M111 S7",
        "M115",
        "M16 Ender 3v2",
        "M300 S110 P50",
        "M118 hello",
        "G91",
        "G0 X-10 Y-10 Z-10",
        "M400",
        "M118 hello",
        "M300 S110 P50",
        "M16 Bonjour",
        "G0 X10 Y10 Z10",
        "M400",
        "M300 S110 P50",
        "M118 END",
    ]
    for m in messages:
        sleep(0.1)
        print("cmd: ", m)
        send_msg(ser, m)
    ENDED = False
    N = 0
    while not ENDED and N < 1000:
        for ans in read(ser):
            print("ans: ", ans)
            if "END" in ans:
                ENDED = True
        sleep(0.1)
        N += 1
    ser.close()
