from sys import maxsize as MAX_INT
import socket
import getpass

SEED0 = 2424

HOSTNAME = socket.gethostname()
SERVER = HOSTNAME.split(".")[0]

match SERVER:
    case "server name":
        DATA_PREFIX = "root path to data on server"
    case _:
        raise ValueError(f"Unknown server: {SERVER}")

USERNAME = getpass.getuser()
match USERNAME:
    case "user name":
        OUTPUT = "path to outputs"
    case _:
        raise ValueError(f"Unknown user: {USERNAME}")
    
class Paths:
    coal = f"{DATA_PREFIX}/Coal-Dust-Explosion"
    blast = f"{DATA_PREFIX}/Circular-Blast"
    output = OUTPUT

class CoalConstants:
    trajlen = 350 - 1
    sim_time = 0.003
    task = "Coal"

    xresolution = 520
    yresolution = 104
    xlength = 0.25
    ylength = 0.05
    xspacing = xlength / xresolution
    yspacing = ylength / yresolution

    pressure = 'pressure'
    volume_fraction_coal = 'volume_fraction_coal'
    xVel_gas = 'xVel_gas'
    yVel_gas = 'yVel_gas'
    temperature_gas = 'temperature_gas'


class BlastConstants:

    trajlen = 50 - 1
    sim_time = 0.005
    task = "Blast"

    xresolution = 128
    yresolution = 128
    xlength = 1.
    ylength = 1.
    xspacing = xlength / xresolution
    yspacing = ylength / yresolution

    pressure = 'pressure'
    density = 'density'
    xVel = 'xVel'
    yVel = 'yVel'
    temperature = 'temperature'