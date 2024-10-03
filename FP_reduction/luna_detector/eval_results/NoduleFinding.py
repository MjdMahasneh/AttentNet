class NoduleFinding(object):
    '''
    Represents a nodule
    '''

    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
                 CADprobability=None, noduleType=None, diameter=None, state=None, seriesInstanceUID=None):
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.noduleType = noduleType
        self.diameter_mm = diameter
        self.state = state
        self.candidateID = None
        self.seriesuid = seriesInstanceUID
