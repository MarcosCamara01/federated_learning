import torch
from model import SimpleCNN
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

global_model = SimpleCNN()

@app.route('/get_model', methods=['GET'])
def get_model():
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    return buffer.getvalue()

@app.route('/update_model', methods=['POST'])
def update_model():
    client_model = torch.load(io.BytesIO(request.data))
    for key in global_model.state_dict().keys():
        global_model.state_dict()[key] = (global_model.state_dict()[key] + client_model[key]) / 2
    return "Model updated"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
