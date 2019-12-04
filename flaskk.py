from flask import Flask, request, jsonify
app = Flask(__name__)
from main2 import *
from mm import *

@app.route('/count',methods =['POST'])
def count_vehicles():
    request_data = request.get_json()
    if "file_path" in request_data.keys():
        cc = cut(request_data['file_path'],request_data['line'],request_data['graph_path'],request_data['time'])
        status = cc.convert(request_data['file_path'],request_data['time'])

        var = Count(request_data['file_path'],request_data['line'],request_data['graph_path'])
        count = var.vehicle()
      
        if status:
            return jsonify("success ",di)
        else:
            return jsonify("Not Success")
    else:
        return jsonify("FilePath Not Given")
        
if __name__ == '__main__':
    app.run(debug= True)

