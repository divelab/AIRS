"""
This file is intended as an example of setting up an ALIGNN model to listen
for and respond to prediction requests using ZMQ.

A minimal set of dependencies required for this example would be:
Python 3.9
pyzmq = 25.1.1
alignn = 2023.8.1

It may be best, and even necessary, to create a separate Python environment
for this process when interoperating with a process running the MCTS code in
this repository, for example, since there could be conflicting dependency
requirements.
"""
import zmq
# noinspection PyUnresolvedReferences
from jarvis.core.atoms import Atoms
# noinspection PyUnresolvedReferences
from alignn.pretrained import get_prediction


if __name__ == '__main__':
    device = "cpu"
    model_name = "jv_formation_energy_peratom_alignn"
    zmq_port = 5555

    print(f"using device: {device}")
    print(f"using model: {model_name}")
    print(f"using port: {zmq_port}")

    # Prepare the ZeroMQ context and REP socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{zmq_port}")

    print("listening for requests...")

    while True:
        # Wait for the next request from the client
        message = socket.recv_string()
        print(f"received request: {message}")

        try:

            try:
                atoms = Atoms.from_cif(from_string=message, get_primitive_atoms=True)
            except Exception as e:
                print(f"exception getting atoms: {e}")
                print("trying with get_primitive_atoms=False...")
                atoms = Atoms.from_cif(from_string=message, get_primitive_atoms=False)

            out_data = get_prediction(
                model_name,
                device,
                atoms=atoms,
                cutoff=8,
                max_neighbors=12
            )

            reply = f"{out_data[0]}"

        except Exception as ex:
            print(f"exception making prediction: {ex}")
            reply = "nan"

        print(f"sending reply: {reply}")
        socket.send_string(reply)
