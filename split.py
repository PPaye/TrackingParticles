'''
Description:
Splitting the Data
You may want to split the training sample to cross validate your algorithm. However, you can't split the sample randomly as the hits originating from the same track mu st be kept together. These functions will split the data such that the samples can be used to cross-validate the algorithm using the scoring function provided below. 
'''
def Split_frac(df, frac):
    '''
    df: Dataframe that will be split
    
    frac: Fraction of data you want to run your algorithm over 
    '''
    
    # Maximum number of tracks 
    m = df['particle_id'].max()  
    split = int(frac*m)
    
    # Want to train algorithm using df1 and cross-validate using df2
    df1 = df.query(f'particle_id<{split}')
    df2 = df.query(f'particle_id>{split}')
    return df1, df2
# 
# Another kind to split the data 
# The definition is hit_id
def split_frac(df, fraction):
    '''
    df: Dataframe that will be split
    frac: Fraction of data you want to run your algorithm over 
    '''
    
    # Maximum number of tracks 
    m = df['hit_id'].max() 
    #print("error", m)                
    split = int(fraction*m)        
    #print("error", m)          
        
    # Want to train algorithm using df1 and cross-validate using df2
    df1 = df.query(f'hit_id<{split}') #.copy(deep=True)
    df2 = df.query(f'hit_id>{split}') #.copy(deep=True)
    return df1, df2


def Split_N(df, n):
    '''
    n: Number of samples you want to split into 
    '''
    grouped = df.groupby(['particle_id'])
    group = np.array_split(grouped, n)
    df = []
    for j in range(n):
        # array of dataframes containing the split of the original data
        df.append(pd.concat([pd.DataFrame(group[j][i][1]) for i in range (len(group[j]))]))
    
    return df
'''Example 
#df_train = Split_N(df, 3)
df,_ = Split_frac(df,0.004)
df.shape
'''