import numpy as np
import pandas as pd
import math
import multiprocessing
import copy

class MarketClearing:
    def __init__(self,inHours, inMaxPrice, inMinPrice, inBucketType, inBucketSize, inBidCollection):

        self.Hours = inHours
        self.MaxPrice = inMaxPrice
        self.MinPrice = inMinPrice
        self.BucketType = inBucketType
        self.BucketSize = inBucketSize
        self.BidCollection = inBidCollection
        self.BidStack = np.zeros([self.MaxPrice-self.MinPrice+1,self.Hours],dtype=np.float64)
        self.OfferStack = np.zeros([self.MaxPrice-self.MinPrice+1,self.Hours],dtype=np.float64)
        self.BidCurve = np.zeros([self.MaxPrice-self.MinPrice+1,self.Hours],dtype=np.float64)
        self.OfferCurve = np.zeros([self.MaxPrice-self.MinPrice+1,self.Hours],dtype=np.float64)
        self.BidOfferCurve = np.zeros([self.MaxPrice-self.MinPrice+1,self.Hours],dtype=np.float64)
        self.InitialMCP = np.zeros(self.Hours)

        self.PriceIndex = np.asmatrix(range(self.MinPrice,self.MaxPrice+1))

        print("Initialised MarketClearing Object")

        self.MakeBidOfferCurve()

        self.MakeSurplusRanges()

        self.MaximizeSurplus()

    def MaximizeSurplus(self):

        threads = 1
        logevery = 10000000
        combis = 2 ** len(self.BidCollection.BlockOffers)

        perthread = int(combis/threads)

        jobs = []
        for i in range(0,threads):
            p = multiprocessing.Process(target=self.MaximiseSurplusProcess, args=(i*perthread,i*perthread+perthread,len(self.BidCollection.BlockOffers),i,logevery,))
            jobs.append(p)
            p.start()

        p = multiprocessing.Process(target=self.MaximiseSurplusProcess, args=(i * perthread, combis, len(self.BidCollection.BlockOffers),i, logevery,))
        jobs.append(p)
        p.start()

        for i in jobs:
            i.join()

    def MaximiseSurplusProcess(self,Start,End, length, Worker, logevery):

        MaxSurplus = 0

        for i in range(Start,End):

            self.BidOfferCurve = np.zeros([self.MaxPrice - self.MinPrice + 1, self.Hours], dtype=np.float64)

            bitstr = str(bin(i))[2:].zfill(length)
            offersaccepted = np.asarray(list(map(int, list(bitstr)))).flatten()

            BlocksAccepted = (offersaccepted[:, np.newaxis] * self.BidCollection.BlockOfferVolumes).sum(axis=0)

            OfferCurve = self.ShiftOffers(BlocksAccepted)

            BidOfferCurve = OfferCurve - self.BidCurve

            MCP = self.FindMCP(BidOfferCurve)

            Surplus = self.CalculateSurplus(MCP,BlocksAccepted)

    def CalculateSurplus(self,MCP,Blocks):

        # the additional consumer surplus on the left
        surplus = Blocks * MCP
        surplus2 = list(map(sum, self.PS))

        return surplus

    def ShiftOffers(self,inBlocksAccepted):

        OfferCurve = copy.deepcopy(self.OfferCurve)

        for i in range(0,len(OfferCurve)):
            OfferCurve[i] += inBlocksAccepted

        return OfferCurve

    def MakeSurplusRanges(self):

        self.SetProducerSurplusRange()
        self.SetConsumerSurplusRange()
        print("Made Surplus Ranges")

    def SetProducerSurplusRange(self):

        self.PSStart = []
        self.PSEnd = []
        self.PSLength = []
        self.PS = []

        for p in range(0, self.Hours):

            self.PSStart.append([])
            self.PSEnd.append([])
            self.PSLength.append([])
            self.PS.append([])

            i = self.MinPrice

            while (i < self.MaxPrice):

                tmpVol = self.OfferCurve[i][p]
                self.PSStart[p].append(i)

                while (i < self.MaxPrice and self.OfferCurve[i][p] == tmpVol):
                    i += 1

                self.PSEnd[p].append(i)
                self.PSLength[p].append(self.PSEnd[p][-1] - self.PSStart[p][-1])
                self.PS[p].append(self.PSLength[p][-1]*tmpVol)

        print("Made PS Range")

    def SetConsumerSurplusRange(self):

        self.CSStart = []
        self.CSEnd = []
        self.CSLength = []

        for p in range(0, self.Hours):

            self.CSStart.append([])
            self.CSEnd.append([])
            self.CSLength.append([])

            i = self.MaxPrice

            while (i > self.MinPrice):

                tmpVol = self.BidCurve[i][p]
                self.CSEnd[p].append(i)

                while (i > self.MinPrice and self.BidCurve[i][p] == tmpVol):
                    i -= 1

                self.CSStart[p].append(i)
                self.CSLength[p].append(self.CSEnd[p][-1] - self.CSStart[p][-1])


        print("Made CS Range")


    def FindMCP(self,inBidOfferCurve):

        MCP = np.zeros(self.Hours)

        for i in range(0,self.Hours):
            MCP[i] = self.find_zero(inBidOfferCurve[:,i])#np.searchsorted(self.BidOfferCurve[:,i], 0, side="left")

        return MCP

    def CalculateTotalSurplus(self):

        #for each period
        for p in range(0,self.Hours):
            p=p


        print("Calculated Surplus")
    def CalculateConsumerSurplus(self):

        print("Calculated Consumer Surplus")
    def MakeBidOfferCurve(self):

        for i in self.BidCollection.BidOfferList:
            if(i.Volume < 0):
                ix = int(i.Price) - self.MinPrice
                self.BidStack[ix,i.HourID] += i.Volume
            else:
                ix = int(i.Price) - self.MinPrice
                self.OfferStack[ix,i.HourID] += i.Volume

        it = np.nditer(self.OfferStack, flags=['multi_index'])
        for i in it:
            if(it.multi_index[0]>0):
               self.OfferCurve[it.multi_index] += self.OfferCurve[it.multi_index[0]-1,it.multi_index[1]]+self.OfferStack[it.multi_index[0]-1,it.multi_index[1]]

        for i in range(self.MaxPrice - self.MinPrice-1,-1,-1):
            for j in range(0,self.Hours):
                self.BidCurve[i,j] += self.BidCurve[i+1,j]-self.BidStack[i,j]

        print("Made BidOffer Curve")

    def find_zero(self, array):
        value = 0
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

class BidCollection:
    def __init__(self):

        self.BidOfferList = []
        self.BlockOffers = []

        print("Initialised Bid Collection")

    def LoadFromText(self,FolderPath):
        print("loading From Bids from Text in " + FolderPath)
        aOfferPrice = np.atleast_2d(np.loadtxt(FolderPath + "/OfferPrice.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))
        aOfferVolume = np.atleast_2d(np.loadtxt(FolderPath + "/OfferVolume.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))
        aBlockOffers = np.atleast_2d(np.loadtxt(FolderPath + "/BlockOffers.txt", np.float64, delimiter='\t', skiprows=1, usecols=range(1, 26)))
        aBidPrice = np.atleast_2d(np.loadtxt(FolderPath + "/BidPrice.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))
        aBidVolume = np.atleast_2d(np.loadtxt(FolderPath + "/BidVolume.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))

        it = np.nditer([aBidPrice,aBidVolume], flags=['multi_index'])

        for x, y in it:
            tmpBid = HourlyBid(it.multi_index[0],it.multi_index[1],x,-y)
            self.BidOfferList.append(tmpBid)

        it = np.nditer([aOfferPrice,aOfferVolume], flags=['multi_index'])
        for x, y in it:
            tmpBid = HourlyBid(it.multi_index[0],it.multi_index[1],x,y)
            self.BidOfferList.append(tmpBid)

        for row in aBlockOffers:
            tmpBlk = BlockOffer(0,row[0],row[1:])
            if(tmpBlk.TotalVolume > 0):
                self.BlockOffers.append(tmpBlk)

        self.BlockOfferVolumes = np.zeros([len(self.BlockOffers),24])

        for i in range(0,self.BlockOfferVolumes.shape[0]):
            for j in range(0, self.BlockOfferVolumes.shape[1]):
                self.BlockOfferVolumes[i,j] = self.BlockOffers[i].Volume[j]

        print("Loaded From Text")


class BlockOffer:
    def __init__(self, inOwnerID, inPrice, inVolume):
        self.OwnerID = inOwnerID
        self.Price = inPrice
        self.Volume = inVolume
        self.TotalVolume = np.sum(inVolume)

class HourlyBid:
    def __init__(self, inOwnerID, inHour, inPrice, inVolume):

        self.OwnerID = inOwnerID
        self.HourID = inHour
        self.Price = inPrice
        self.Volume = inVolume
        print("Initialised Hourly Bid")
