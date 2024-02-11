import pickle
import sys
import numpy
import pandas
import sklearn

modelPath='modelo_regresion_logistica.pkl'
scalerPath='escalador.pkl'


with open(scalerPath, 'rb') as file:
    scaler = pickle.load(file)

with open(modelPath, 'rb') as file:
    model = pickle.load(file)

if len(sys.argv)>0:
  arguments = sys.argv[1:]

  nuevos_datos = numpy.array([float(arg) for arg in arguments])
  nuevos_datos = numpy.expand_dims(nuevos_datos, axis=0)
  nuevos_datos_scaler = scaler.transform(nuevos_datos)
  prediccion = model.predict(nuevos_datos_scaler)

  print('Prediccion', prediccion)

else:
   print('No se recibieron argumentos')
