import numpy as np
import pandas as pd
import math

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
        self.FindMCP()
    def FindMCP(self):

        for i in range(0,24):
            self.InitialMCP[i] = self.find_zero(self.BidOfferCurve[:,i])#np.searchsorted(self.BidOfferCurve[:,i], 0, side="left")

        print(self.InitialMCP)

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

        self.BidOfferCurve =  self.OfferCurve - self.BidCurve

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

        print("Initialised Bid Collection")

    def LoadFromText(self,FolderPath):
        print("loading From Bids from Text in " + FolderPath)
        aOfferPrice = np.atleast_2d(np.loadtxt(FolderPath + "\\OfferPrice.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))
        aOfferVolume = np.atleast_2d(np.loadtxt(FolderPath + "\\OfferVolume.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))
        aBidPrice = np.atleast_2d(np.loadtxt(FolderPath + "\\BidPrice.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))
        aBidVolume = np.atleast_2d(np.loadtxt(FolderPath + "\\BidVolume.txt", np.float64,delimiter='\t',skiprows=1,usecols=range(1,25)))

        it = np.nditer([aBidPrice,aBidVolume], flags=['multi_index'])
        for x, y in it:
            tmpBid = HourlyBid(it.multi_index[0],it.multi_index[1],x,-y)
            self.BidOfferList.append(tmpBid)

        it = np.nditer([aOfferPrice,aOfferVolume], flags=['multi_index'])
        for x, y in it:
            tmpBid = HourlyBid(it.multi_index[0],it.multi_index[1],x,y)
            self.BidOfferList.append(tmpBid)

        print("Loaded From Text")


class HourlyBid:
    def __init__(self, inOwnerID, inHour, inPrice, inVolume):

        self.OwnerID = inOwnerID
        self.HourID = inHour
        self.Price = inPrice
        self.Volume = inVolume
        print("Initialised Hourly Bid")
