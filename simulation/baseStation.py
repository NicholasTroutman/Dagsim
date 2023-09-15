#Base Station Nodes Immobile and only nodes with wider-internet networking
from agent.py import Agent

class BaseStation(Agent)
	def __init__(self, _counter, _position):
		Node.__init__(self, _counter)
		self.position = _position


		#self.coordinates= [] #list of double x and y coordinates in double [x,y]
		#self.radius=60 #hardcoded radius of p2p connectivity

        ##block variables
        #self._visible_transactions=[]
        #self.confirmed_transactions=[]
        
        ##transaction variables
        #self._visible_blocks = []
        #self.confirmed_blocks = []
        
        ##For analysis
        #self.agent_average_confirmation_confidence = 0
        #self.tips = []
        #self.record_tips = []


