import numpy as np

def Scoring(particles, reconstructed_tracks): 
    # Only those particles with greater than 3 hits should be reconstructible
    for i, track in particles.items():
        if (len(track) <3):
            particles.drop(labels=i, inplace=True)
    
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
            #if np.array_equal(particle, track):
            #   good_tracks=good_tracks+1
        
            if ((len(np.intersect1d(track, particle))/ len(particle))>=0.7):
                # np.intersect1d returns the sorted, unique values that are in both of the input arrays. 
                good_tracks=good_tracks+1  
                clones = clones+1
        if clones>1:
            cloned = cloned+(clones-1) 
            # Only count one of the good tracks with good overlap
            good_tracks = good_tracks - (clones-1)
    try:
        efficiency = good_tracks/len(particles)  
        fake = (len(reconstructed_tracks)-good_tracks)/len(reconstructed_tracks)
        clone_rate = cloned/len(reconstructed_tracks)
    except ZeroDivisionError:
        return {'efficiency':0, 'fake_rate':0, 'clone_rate':0}
    
    return {'efficiency':efficiency, 'fake_rate':fake, 'clone_rate':clone_rate}
