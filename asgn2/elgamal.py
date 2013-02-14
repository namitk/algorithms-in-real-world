# using finitefield library from http://userpages.umbc.edu/~rcampbel/Computers/Python/numbthy.html#Packages
from finitefield import FiniteField, FiniteFieldElt

class elgamal:
    dictionary = {}
    rev_dictionary = {}
    private_key = 1
    irred_poly = []

    def __init__(self, coeff_group, irred_poly, private_key):
        self.irred_poly = irred_poly
        self.private_key = private_key

        def count(counter, upper_limits):                
            n = len(counter)
            while True:
                index = 0
                while index < n:
                    counter[index] += 1
                    if counter[index] < upper_limits[index]:
                        return counter[:]                
                    counter[index] = 0
                    index += 1
                    if index > n-1:
                        return

        counter=[0]*3
        upper_limits=[coeff_group]*3
        letters ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'        
        for i in range(26):
            self.dictionary[letters[i]]=count(counter, upper_limits)
        self.rev_dictionary = dict((str(v),k) for k, v in self.dictionary.iteritems())
        self.GF = FiniteField(coeff_group, irred_poly)
        
    def decode(self, encoded_char):
        y1 = encoded_char[0]
        y2 = encoded_char[1]
        
        y1poly = FiniteFieldElt(self.GF, self.dictionary[y1])
        y2poly = FiniteFieldElt(self.GF, self.dictionary[y2])     
        
        inv_y1_raised_to_private_key = (y1poly**self.private_key)**25

        ans = eval(str(y2poly*inv_y1_raised_to_private_key))
        assert(len(ans)==3)        
        return self.rev_dictionary[str(ans)]

coeff_group = 3
private_key = 131
irred_poly = [1, 0, 2] #leading coefficient assumed to be 1 by the library
message = [('P', 'D'), ('J', 'P'), ('K', 'K'), ('D', 'O'), ('O', 'O'), ('X', 'Y'), ('P', 'O'), ('R', 'P'), ('I', 'Y'), ('S', 'D'), ('Y', 'G'), ('Z', 'T'), ('X', 'M'), ('F', 'O'), ('L', 'L'), ('Z', 'I'), ('E', 'W'), ('V', 'L'), ('B', 'K'), ('T', 'N'), ('S', 'D'), ('Z', 'T'), ('X', 'U'), ('X', 'G'), ('X', 'M'), ('E', 'Y'), ('V', 'X'), ('X', 'D'), ('V', 'R'), ('B', 'I'), ('V', 'V'), ('Y', 'V'), ('T', 'S'), ('Z', 'E'), ('E', 'L'), ('D', 'S')]
eg = elgamal(coeff_group, irred_poly, private_key)
s=""
for elem in message:
    s+=eg.decode(elem)    
print s
