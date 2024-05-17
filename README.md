
1.originTracks,validTracks,validIndex = loadCoordinateFromCSV(filepath='/ldap_shared/home/v_yy/Project/MoTT/cluster/test4/',filename='test.csv')
   The purpose of the loadCoordinateCSV.py file is to cluster the particles and draw their boundaries, calculate the area, and count the number of particles
   The function of the loadCoordinateFromCSV function is to initialize the csv file, and arrange all particles in ascending order according to the frame after grouping according to the track_id.
   The input parameters of the function are the path and file name of the csv file.
   There are three return values for the function, which are the original record of all particles after grouping and sorting, and the particles containing more than 15 consecutive frames and their indexes

2.dataByFrame,maxFrameNum = getCoordinateByFrame(validTracks)
   The purpose of the getCoordinateByFrame function is to get a set of candidate particles by frame classification

3.clusterData = cluster(dataByFrame,maxFrameNum)
   The purpose of the cluster function is to cluster candidate particles

4.plotParticleEveryFrameInCluster(maxFrameNum,dataByFrame,clusterData,filepath='/ldap_shared/home/v_yy/Project/MoTT/cluster/test4/')
   The purpose of the plotParticleEveryFrameInCluster function is to plot the boundaries and calculate the enclosed area of the clustered particles
