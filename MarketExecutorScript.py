import MarketClearing as MC
from inspect import getsourcefile
from os.path import abspath
from os.path import dirname

if __name__ == '__main__':
    oBC = MC.BidCollection()

    myPath = abspath(getsourcefile(lambda:0))
    myPath = dirname(myPath)
    oBC.LoadFromText(myPath)
    oMC = MC.MarketClearing(24,500,0,0,500,oBC)

    print("All Done")
