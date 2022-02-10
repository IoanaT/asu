Hey,

finally technology requirements functioneaza, au fost ceva probleme. what I've learned:

- aveai instalat python3.6 which is enough, am dezinstalat python3.7
- cand scriai in linia de comanda $ python, rula python2.something, am schimbat sa ruleze python3.6
- ti am instalat si git
- in /home/ioana/asu ai /pgmpy de pe github si fisierul bn.py de pe coursera
- in /pgmpy era in fisierul requirements.txt era o librarie careia i s a schimbat intre timp numele, din pytorch in torch, dupa ce am schimbat asta a putut sa instaleze tot, daca mai rulezi comanda din "technology requirements" o sa zica ca ai toate instalate
- legat de bayesian thingy s a schimbat ceva in pgmpy si din cauza asta nu mergea cand executai $ python bn.py, o trebuit sa schimb in fisier direct, am adaugat joint=False:

# Query
print(cancer_infer.query(variables=['Dyspnoea'], evidence={'Cancer': 0},joint=False)['Dyspnoea'])
print(cancer_infer.query(variables=['Cancer'], evidence={'Smoker': 0, 'Pollution': 0},joint=False)['Cancer'])

aici e linkul cu issue ul respectiv: https://github.com/pgmpy/pgmpy/issues/1111

- cand rulezi python bn.py iti da asemanator cu ce scrie in hartii

SPOR mai departe, ca mai departe eu nu ma mai stiu !

p.s. text editor poti sa folosesti gedit din linia de comanda sau right click si open with text editor

