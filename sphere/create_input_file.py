import yaml

d = {'Geometry/Mesh':{'xmin': -0.9873862,\
                      'ymin': -0.9867069,\
                      'zmin': -0.9869933,\
                      'xmax':  0.9886856,\
                      'ymax':  0.987431,\
                      'zmax':  0.988227,\
                      'Nx':  1004,\
                      'Ny':  1004,\
                      'Nz':  1004,\
                      'type': 'pec',\
                      'report': True},\
     'Frequency':{'fmin': 0.05,\
                  'fmax': 0.5,\
                  'df': 0.015,\
                  'Nf': 31},\
     'NeuralNetwork':{'ValData': 15,\
                      'LRate': 0.0005,\
                      'ActF': 'lrelu',\
                      'adaptActF':'gaaf',\
                      'nL' : 5,\
                      'bHN' : 32,\
                      'tHN' : 64,\
                      'Ldim': 200,\
                      'Nepoch': 25000,\
                      'Nprint': 500,\
                      'NormMethod':'minmax',\
                      'Optimizer':'adam',\
                      'ModelName': 'E_',\
                      'Acceleration':'gpu',\
                      'Plotting':False}}
                      



with open('input_file.yaml', 'w') as yaml_file:
    yaml.dump(d, yaml_file, default_flow_style=False)
