#Makes Map from grayscale image with red roads and blue intersections
import sys
import timeit
import random
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import rand
from operator import add


## Functions to create graph of greyscale traffic image
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from PIL import Image
import math

#returns Euclidean distance between two 2d poins
def Distance(n1,n2):
    return math.hypot(n1[0]-n2[0], n1[1]-n2[1])


#distance of c to a and b line
def DistanceToVector(a,b,c): 
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    return np.abs(np.cross(b-a, c-a) / np.linalg.norm(b-a))
    
    
#finds red intersections
def FindCenter(image, y, x):
    x2=x-20
    reds=[]
    
    #find center of red dot
    for i in range(0,40):
        for j in range(0,40):
            try:
                if image[y+j][x2+i][0]>100+image[y+j][x2+i][1]: #Red number found
                    reds.append([y+j,x2+i])     
            except IndexError:
                #print("OOB")
                pass
                
        
    result=np.mean(reds, axis=0)
    #print(result)
    
    return [int(result[0]), int(result[1])]



#find blue lines around an intersection
def FindEdges(image, intersections):
    #start at intersection
    allFinal=[] #[[index,[blues]]]
    for index, yx in intersections:
        #print(index)
        #print(yx)
                
        #trace rectangle around
        y2=yx[0]-50
        x2=yx[1]-50
        blues=[]
        
        for i in [0,100]:
            for j in range(0,100):
                try:
                    #print("y: ",y2+j," x: ",x2+i)
                    if image[y2+j][x2+i][2]>100+image[y2+j][x2+i][1]: #blue number found
                        blues.append([y2+j,x2+i])     
                except IndexError:
                    #print("OOB")
                    pass
            
        for i in range(0,100):
            for j in [0,100]:
                try:
                    if image[y2+j][x2+i][2]>100+image[y2+j][x2+i][1]: #Blue number found)
                        blues.append([y2+j,x2+i])     
                except IndexError:
                    #print("OOB")
                    pass        
                
        #get rid of similar blues
        #print(blues)
        final=[]
        while len(blues)>0: #O(n**2)
            same=[]
            same.append(blues[0])
            for b in blues:
                if Distance(blues[0],b)<5: #too similar
                    same.append(b)
            
            final.append(np.mean(same, axis=0))
            blues=[b for b in blues if b not in same]
            #print(blues)
        
        #print(final)
        
        finalPretty= [[int(x),int(y)] for x,y in final]
        allFinal.append([index,finalPretty])
        
        
        #pass all blues to IdentifyBlueEdgeIntersections
        neighbors=[] #[[index, [neighbpors]]]
        for index, blues in allFinal:
            n=IdentfiyBlueEdgeIntersection(intersections, intersections[index],blues)
            neighbors.append([index,n])
        
    return neighbors #integer response



#find which intersection the edge points to given intesection, bluepoint, and other intersections
def IdentfiyBlueEdgeIntersection(intersections, intersection, bluepoint ):
    #
    neighbors=[] #[index, distance]
    
    
    for bp in bluepoint: #for each blue edge
        #print("\nblueedge: ",bp)
        
        closestIntersection=[-1, 9999999999999999]
        
        for index, yx  in intersections: #for each other intersections
            
            if index!=intersection[0]: #not same as origin
                if DistanceToVector(intersection[1], bp, yx) < 50: #close enough to match
                    distance=Distance(yx,intersection[1]) #measure distance
                    #print("\n")
                    #print(index)
                    #print("Distance: ",distance)
                    #print("D to V: ",DistanceToVector(intersection[1], bp, yx))
                    #print("CI: ",closestIntersection[1])
                   
                    
                    if Distance(yx, bp) < Distance(yx,intersection[1]):#correct direction
                        #print("Correct Direction")
                        
                        if distance<closestIntersection[1]: #closest to originating intersection
                            closestIntersection=[index, distance]
                
        neighbors.append(closestIntersection)
        #print("\nNeighbors: ",neighbors)
    return neighbors


##take image, return full graph with nodes + edges
def LoadImageIntoGraph(image):
    im_frame = Image.open(image)
    np_frame = np.array(im_frame.getdata())
    np_frame.shape=(int(np_frame.shape[0]/im_frame.width), im_frame.width,  3)

    #plt.imshow(np_frame)
    #plt.show()

    intersections=[]
    for y in range(0,np_frame.shape[0]):
          for x in range(0,np_frame.shape[1]):  
              if np_frame[y][x][0]>np_frame[y][x][1]+100:
                  flag=True
                  #
                  #check if new point is too close existing intersections
                  for index, yx in intersections:
                      if Distance(yx,[y,x]) <40:
                         flag=False
                 
                  if flag:
                      intersections.append([len(intersections),FindCenter(np_frame,y,x)])
        
    #for i in range(0,len(intersections)):
    #    print(i,": ",intersections[i][1])          
                
    ##plot intersections
    #plt.imshow(np_frame)
    #for index, yx in intersections:
    #    plt.scatter(yx[1],yx[0],s=40,color="green")
    #    plt.annotate(index,[yx[1],yx[0]])
    #plt.show()

    #get edges
    edges=FindEdges(np_frame, intersections) #Edges= [[index, [neighbors]]] #neighbors=[index,distance]
    
    G = nx.Graph()

    #create nodes
    for index, yx in intersections:
        G.add_node(index,pos=(yx[1],yx[0]))
        
    #create edges
    for index, edge in edges:
        for index2, distance in edge:
            if index2>index: ##for only 1 edge between
                G.add_edge(index,index2,weight=int(distance))

    
    return G, np_frame #graph, image


##https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment
def isBetween(a, b, c):
    #print("\nisBetween:")
    #print(a)
    #print(b)
    #print(c)
    
    if (abs(Distance(a,b)+Distance(b,c)-Distance(a,c))<1): #epsiolon=1
        #print("IS BETWEEN")
        return True
    #print("NOT BETWEEN")
    return False
