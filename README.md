## RAMP VELO Challenge

Data challenge to reconstruct tracks from hits in the VELO using spatial and time coordinates. Thanks to the team at Paris-Saclay Center for Data Science who developed the RAMP framework which has been used in this challenge.

### Setup
1. Clone the repository
   
   ```bash
   git clone https://gitlab.cern.ch/shtaneja/ramp-velo-challenge-.git
   cd ramp-velo-challenge-
   ```
2. setup a virtual environment or use conda
   
   * with `conda`
   
   ```bash
   conda update conda
   conda env create -f environment.yml
   conda activate ramp_velo_challenge
   
   ```
   
   *  wit `pip` (recommend to use a virtual environment)
   
   ```bash
   python -m pip install -r requirements.txt
   ```

 3. download the data sets
    
    ```bash
    cd data
    python ../download.py
    ```
 4. Launch the Jupyter Notebook to develop your own solution to the problem

 5. Once you are happy with the score of your algorithm, you can upload the list of Tracks, as well as the code you used to find the list of tracks. 
    A naming convention needs to be followed when submitting your reconstructed Tracks. The name of the text file should begin with the detector configuration of the relevant dataset followed by your name i.e. for the dual technology dataset, the file could be called```55microns_50psInner_200microns_50psOuter_Name.txt```. The reconstructed Tracks can be uploaded to the following link:
https://cernbox.cern.ch/index.php/s/y9riDHYUFUtGLRm

 6. The code used to generate the reconstructed Tracks can be uploaded to the following link:
https://cernbox.cern.ch/index.php/s/QAYTsj9Wo9CLZVH
Remember to give your file an appropriate name which includes your name and the type of algorithm used i.e. ```Name_neural_network.py```  
