import numpy as np
from elemNumDir import num_to_elem


class utils_read_write:
    #TODO paar errors abfangen 
    
    def __init__(self):
        pass

    def read_column(self, l_path):
        '''
        Input: Path to label file as txt, single entry per line
        Output: Parsed energies as list
        '''
        e_file = open(l_path, "r")
        labels = []

        for line in e_file:
            labels.append(float(line[:-1]))
        e_file.close()

        return labels

    def readXYZs(self, filename):
        infile = open(filename,"r")
        coords = [[]]
        elements = [[]]
        for line in infile.readlines():
            if len(line.split()) == 1 and len(coords[-1]) != 0:
                coords.append([])
                elements.append([])
            elif len(line.split()) == 4:
                elements[-1].append(line.split()[0].capitalize())
                coords[-1].append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        infile.close()
        return coords,elements


        return coords, elements
    
    def read_npy(self, filename):
        labels = np.load(filename)
        return labels

    def export_xyz(self, coords, elements, filename):
        outfile = open(filename, "w")

        for molidx, mol in enumerate(coords):
            outfile.write("%i\n\n" % (len(mol)))
            
            for atomidx, atom in enumerate(mol):
                outfile.write("%s %f %f %f\n" % (elements[molidx][atomidx], atom[0], atom[1], atom[2]))

    def export_xyz_b(self, coords, elements, filename):
        outfile = open(filename, "w")

        for molidx, mol in enumerate(coords):
            outfile.write("%i\n\n" % (len(mol)))
            
            for atomidx, atom in enumerate(mol):
                outfile.write("%s %f %f %f\n" %(num_to_elem[int(elements[molidx][atomidx])], atom[0], atom[1], atom[2]))

    def export_xyz_single(self, coord, elements, filename):
        outfile = open(filename, "w")

        outfile.write("%i\n\n" % (len(elements)))

        for atomidx, atom in enumerate(coord):
            outfile.write("%s %f %f %f\n" % (elements[atomidx], atom[0], atom[1], atom[2]))

        outfile.close()
        
    def export_labels(self, labels, filename):
        outfile = open(filename + ".txt", "w")
        for value in labels:
            outfile.write("%f \n" % value)
        outfile.close()
        
        np.save(filename + ".npy", labels)
        
    elem_to_num = {
        "H" : 1,
        "He" : 2,
        "Li" : 3,
        "Be" : 4,
        "B" : 5,
        "C" : 6,
        "N" : 7,
        "O" : 8,
        "F" : 9,
        "Ne" : 10,
    }

    num_to_elem = {
        1 : "H",
        2 : "He",
        3 : "Li",
        4 : "Be",
        5 : "B",
        6 : "C",
        7 : "N",
        8 : "O",
        9 : "F",
        10 : "Ne",
    }
