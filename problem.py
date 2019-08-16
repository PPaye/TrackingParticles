import numpy as np
import pandas as pd
import os
import rampwf as rw
from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types import BaseScoreType
import pickle
# Create a class for the predictions


problem_title = "4D Tracking"

# Need a prediction type
# This is a list of hits belonging to some track

#this the custom workflow for the challenge. All it does is read a file and create an array
#passes this array from training to testing and testing to scoring
workflow = rw.workflows.MyWorkflow()


#It initialize the predicted data with a value.
class TrackPredictions(BasePrediction):
    def __init__(self, y_pred=None, y_true=None, n_samples=None):
        if y_pred is not None:
            self.y_pred = y_pred
        elif y_true is not None:
            self.y_pred = y_true
        elif n_samples is not None:
            self.y_pred = np.empty(n_samples, dtype=object)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')

    def __str__(self):
        return 'y_pred = {}'.format(self.y_pred) # PRINT THE VALUE OF predicted data

    @classmethod
    # combination at the moment dummy implementation
    def combine(cls, predictions_list, index_list=None):

        combined_predictions = cls(y_pred=predictions_list[0].y_pred)
        return combined_predictions

    @property
    def valid_indexes(self):
        return self.y_pred != np.empty(len(self.y_pred), dtype=np.object)
        # return True



#predictions object which is used to create wrapper objects for y_pred
Predictions = TrackPredictions
#change the predicitions and label names. Need to use a prediction class like TrackPredictions
#turn scoring function into a class to be used in RAMP scoring method
def Scoring(particles, reconstructed_tracks):
    for a,b in zip(particles, reconstructed_tracks):
        # sort the arrays numerically
        a.sort()
        b.sort()

    good_tracks=0
    cloned=0
    # iterate over particles
    for particle in particles:
    # iterate over tracks reconstructed
        clones = 0
        for track in reconstructed_tracks:
            # check element wise equality between arrays

            if ((len(np.intersect1d(track, particle))/ len(particle))>=0.7):
                # np.intersect1d returns the sorted, unique values that are in both of the input arrays.
                good_tracks=good_tracks+1
                clones = clones+1
        if clones>1:
            cloned = cloned+(clones-1)

    efficiency = good_tracks/len(particles)
    fake = (len(reconstructed_tracks)-good_tracks)/len(reconstructed_tracks)
    clone_rate = cloned/len(reconstructed_tracks)
    return {'efficiency':efficiency, 'fake_rate':fake, 'clone_rate':clone_rate}


#this class contains the scoring methods
#also can make checks for instance do the participants have tracks with only 1-2 hits?
class TrackChecker:
    def __init__(self):
        self.good_tracks = 0
        self.cloned = 0
       
        self.efficiency = 0.0
        self.fake = 0.0
        self.clone_rate = 0.0
        self.tracks = []

    #function to obtain 2d array of tracks (disentangled from the mess of RAMP!)
    def extractTracks(self, y_pred):
        
        for i in range(len(y_pred)):
            self.tracks.append(y_pred[i][0])
          
    def Scoring(self):
        path = str(os.environ['RAMPDATADIR'])
        #if time in path:
           #if size in path:
               #df = read_file(name with right time and size)
	# Should read the testing file once we implement properly
	# In testing phase

        df = pd.DataFrame()
        if(os.environ['RAMP_TEST_MODE'] == '1'):
            print("quick test mode!")
            df = pd.read_csv('RAMP_smallTruth.txt',sep=' ')
        else:
            print("Testing mode!")
            try:
                df = pd.read_csv('/home/RAMPAdmin/ramp-board/ramp_deployment/ramp-data/testing/RAMPData_test_admin.txt', sep=' ')
            except:
                print("File doesn't exist! Did you forget the --quick-test flag?")
        df = df.apply(pd.to_numeric, errors='coerce')
        particles = df.groupby(['particle_id'])['hit_id'].unique()

        # iterate over particles
        for particle in particles:
            if len(particle) < 3:
                continue
        # iterate over tracks reconstructed
            clones = 0
            for track in (pd.Series(np.asarray(v) for v in self.tracks)):
               # print(type(track))
                if len(track) < 3:
                    continue
                intersection = np.intersect1d(track, particle)
                # check element wise equality between arrays
                #if np.array_equal(particle, track):
                #   good_tracks=good_tracks+1

                if ((len(intersection)/ len(particle))>=0.7):
                    # np.intersect1d returns the sorted, unique values that are in both of the input arrays.
                    self.good_tracks = self.good_tracks+1

                    clones = clones+1
            if clones>1:
                self.cloned = self.cloned+(clones-1)
        #need a scoring function for the total score such as eff*(1-clone rate)*(1 - fakerate)^2
        #self.total =
        self.efficiency = self.good_tracks/len(particles)
        self.fake = (len(self.tracks)-self.good_tracks)/len(self.tracks)
        self.clone_rate = self.cloned/len(self.tracks)
        #self.metrics = {'efficiency':efficiency, 'fake_rate':fake, 'clone_rate':clone_rate}
        print ("Efficiency: " + str(self.efficiency))
        print ("Fake Rate: " + str(self.fake))
        print ("Clone Rate: " + str(self.clone_rate))

#Class which wraps the checking class for the scoring. Only calls the methods from TrackChecker
class TrackScore_total(BaseScoreType):
    #must have is_lower_the_better! Not sure if minimum and maximum are needed
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    #constructor
    def __init__(self, mode, name='total score', precision=3):
        self.name = name
        self.precision = precision
        self.mode = mode
        self.checker = TrackChecker()
    def __call__(self, y_true_label_index, y_pred_label_index):
        if len( self.checker.tracks)==0:
             self.checker.extractTracks(y_pred_label_index)
             self.checker.Scoring()
        #checker.final_score()
        #data = {'efficiency':self.checker.efficiency, 'fake_rate':self.checker.fake, 'clone_rate':self.checker.clone_rate} 
        data = np.array([self.checker.efficiency, self.checker.efficiency, self.checker.fake, self.checker.clone_rate])
        s = pd.Series(data)          
        return s


#these are the scoring metrics that we want. track
score_types = [
    TrackScore_total(name="total", mode="total"),
    TrackScore_total(name="efficiency", mode="eff"),
    TrackScore_total(name="fake rate", mode="fake"),
    TrackScore_total(name="clone rate", mode="clone")

]


#cv method
#does nothing essentially, training and testing split for training data is up to users discretion

def get_cv(X,y):

    n_tot = len(y)


    
    temp = [(np.r_[0:n_tot], np.r_[0:n_tot])]



    return temp
#I/O functions
#DO NOT DELETE _read_data, get_train_data or get_test_data. RAMP needs these methods.
def _read_data(path):
    
    #read in the user's submission

    # Solution file uploaded by participants
    dataPath = str(path) + 'Tracks.txt'
    os.environ['RAMPDATADIR'] = path



    df =  pd.read_json(dataPath, orient='values', typ='series')

    y = df.values.tolist()

    print("There are " + str(len(y)) + " tracks!")

    return df, y

def get_train_data(path):
    return _read_data(path)

def get_test_data(path):

    return _read_data(path)


def main()
    

            
main()