#se creo un simple api la cual recibira las coordenadas del punto del mapa que necesitamos para poder realizar el metodo de los poligonos
from flask import Flask, jsonify, request,send_file

app = Flask()

if __name__ == '__main__':
    app.run(debug=True, port=8000)


@app.route('/poligonosT', methods = ['GET'])
def hello():
    x = request.args.get('x')
    y = request.args.get('y')
  
    cordenada = x + y

    return cordenada
