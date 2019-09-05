%% Motion-Based Multiple Object Tracking

function multiObjectTracking1()
clc;clear all; close all;
obj = setupSystemObjects();
tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track
frameCount = 0;
%frameData(1)= struct('fishesData',{});

% Detect moving objects, and track them across video frames.
while ~isDone(obj.reader)
    frame = readFrame(); 
    [centroids,bboxes,frame,mX, mY,bX,bY] = detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] =  detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
   % deleteLostTracks();
    createNewTracks();
    
    displayTrackingResults();
end

 save('frameData.mat', 'frameData');

%% Create System Objects
    function obj = setupSystemObjects()
        obj.reader = vision.VideoFileReader('fishC.AVI');
        %obj.reader = vision.VideoFileReader('fish2.avi');
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        obj.maskPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        obj.detector = vision.ForegroundDetector('NumGaussians', 3,     'NumTrainingFrames', 5, 'MinimumBackgroundRatio', 0.4);
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true,    'AreaOutputPort', true, 'CentroidOutputPort', true,   'MinimumBlobArea', 100);
    end

%% Initialize Tracks
    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'kalmanFilter', {}, ...
             'bbox', {}, ...
             'age', {}, ...
             'midlineX',{},...
             'midlineY',{},...
             'boundaryX',{},...
             'boundaryY',{},...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end

%% Read a Video Frame
% Read the next video frame from the video file.
    function frame = readFrame()
        frame = obj.reader.step();
    end

%% Detect Objects
    function [centroids, bboxes,frame1,mX, mY,bX,bY ] = detectObjects(frame)
            fudgeFactor = 0.6;                                              % threshold to detect of fish 
            minclassArea = 150;                                           % variable to express minimum area for noise removing.
            minclassArea1 = 100;                                         % variable to express minimum area of each class, if any class is lower than this value, we remove this class.

            gray_IMG = rgb2gray(frame);
            gray_IMG=imgaussfilt( imadjust(gray_IMG));
            [~, threshold] = edge(gray_IMG, 'sobel');
            
            BWs = edge(gray_IMG,'sobel', threshold * fudgeFactor);

            se90 = strel('line', 2, 90);
            se0 = strel('line', 2, 0);
            BWsdil = imdilate(BWs, [se90 se0]);

            BWnobord = imclearborder(BWsdil, 4);
            BWnobord1 = bwareaopen(BWnobord, minclassArea);
            %imshow(BWnobord1);
           
            BWsdil =BWnobord1;
            BWdfill = imfill(BWsdil, 'holes');

            seD = strel('disk',3);
            BWfinal = imerode(BWdfill,seD);
            BWfinal = imdilate(BWfinal, seD);
            BWnobord = bwareaopen(BWfinal, minclassArea1);
            BWfinal = BWnobord;
            BWoutline = bwperim(BWfinal);
       % imshow(BWoutline);
          %%  display the center position of each fish
            [row, col] = find(BWoutline==1);
            for i = 1:length(row)
                frame(row(i),col(i),:) = [0,255,0];    
            end

            BWnobord = bwmorph(logical(BWfinal),'thin',Inf);
            [row, col] = find(BWnobord==1);
            for i = 1:length(row)
                frame(row(i),col(i),:) =[ 255,0,0];     
            end

            seD = strel('disk',3);
            BWfinal = imerode(BWfinal,seD);
            BWfinal = bwareaopen(BWfinal, 51);
            individual_fish = regionprops(BWfinal,'centroid');
            centroids = cat(1, individual_fish.Centroid);
            bboxes = centroids;
            frame1 = frame;
            
         %% get the center line and boundary of each fish
         
            L = bwlabel(BWfinal,4);
            num =max(max(L));
            mX={};        mY={};        bX={};        bY={};        n = 0;
         for i=1:num
             tmp_Img=logical(zeros(size(L)));
            rc = find(L == i);
            if length(rc) <= 49 
                continue;
            end
            
            tmp_Img(rc) = 1;
            BWnobord = bwmorph(logical(tmp_Img), 'thin', Inf);
            [y, x] = find(BWnobord==1);
            n = n+1;
            mX{n}=x;            mY{n} = y;
            
            BWoutline = bwperim(tmp_Img);
            [y, x]  = find(BWoutline == 1);
            bX{n}=x;             bY{n} =y;
            
         end
    end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the current frame, and update its bounding box accordingly.
    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
                                                                            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
                                                                             % Shift the bounding box so that its center is at the predicted location.
            predictedCentroid = int32(predictedCentroid);
            tracks(i).bbox = [predictedCentroid];
        end
    end

%% Assign Detections to Tracks
    function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment()        
        nTracks = length(tracks);
        nDetections = size(centroids, 1);        
                             % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end        
                    % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] =  assignDetectionsToTracks(cost, costOfNonAssignment);
    end

%% Update Assigned Tracks
    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
                       % Correct the estimate of the object's location
                       % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);            
                            % Replace predicted bounding box with detected
                            % bounding box.
            tracks(trackIdx).bbox = bbox;            
                          % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            tracks(trackIdx).midlineX = mX{detectionIdx};
            tracks(trackIdx).midlineY = mY{detectionIdx};
            tracks(trackIdx).boundaryX = bX{detectionIdx};
            tracks(trackIdx).boundaryY = bY{detectionIdx};
                       % Update visibility.
            tracks(trackIdx).totalVisibleCount =  tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end



%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.
    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible for too many consecutive frames. It also deletes recently created tracks that have been invisible for too many frames overall. 
    function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 50;               % after fish is dispear from windows, it is variable to estimate position of fish by using KalmanFilter
        ageThreshold = 20;                          %  after fish is dispear from windows, it is variable to remove 'track' class of that fish.
                                                                 % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
                                                     % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) |    [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
                                                     % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned detection is a start of a new track. In practice, you can use other cues to eliminate noisy detections, such as size, location, or appearance.
    function createNewTracks()
        centroid1 = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        for i = 1:size(centroid1, 1)            
            centroid = centroid1(i,:);
            bbox = bboxes(i, :);
                                                 % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity',  centroid, [200, 50], [100, 25], 100);
                    %first argument is Motion model, specified as the string 'ConstantVelocity' or 'ConstantAcceleration'. The motion model you select applies to all dimensions. For example, for the 2-D Cartesian coordinate system. This mode applies to both X and Y directions.
                    %second argument is Initial location of object, specified as a numeric vector. This argument also determines the number of dimensions for the coordinate system. For example, if you specify the initial location as a two-element vector, [x0, y0], then a 2-D coordinate system is assumed.
                    %third argument is initialestimateError,Initial estimate uncertainty variance, specified as a two- or three-element vector. third argument means [LocationVariance,VelocityVariance].
                    %forth argument is MotionNoise, deviation ofselectedand actual model, specified as a two- or three-element vector. it is [LocationVariance, VelocityVariance]
                    % fifth argument is Measurement Noise, variance inaccuracy of detected location.

                                                % Create a new track.
            %nextId = length(tracks)+1;
            if length(tracks)< unassignedDetections(i)
                newTrack = struct( 'id', unassignedDetections(i),  'kalmanFilter', kalmanFilter,  'bbox', bbox,  'age', 1,  'midlineX' ,mX{i}, 'midlineY',mY{i},'boundaryX',bX{i},'boundaryY',bY{i},'totalVisibleCount', 1,   'consecutiveInvisibleCount', 0);
                                                % Add it to the array of tracks.
                tracks(end+1) = newTrack;
            else
                   tracks(unassignedDetections(i))=struct( 'id', unassignedDetections(i),  'kalmanFilter', kalmanFilter,  'bbox', bbox,  'age', 1,  'midlineX' ,mX{i}, 'midlineY',mY{i},'boundaryX',bX{i},'boundaryY',bY{i},'totalVisibleCount', 1,   'consecutiveInvisibleCount', 0);
            end
            % Increment the next id.
            %nextId = nextId + 1;
        end
    end

%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID for each track on the video frame and the foreground mask. It then displays the frame and the mask in their respective video players. 

    function displayTrackingResults()
        frame = im2uint8(frame);
       
        minVisibleCount = 0;
        if ~isempty(tracks)
            reliableTrackInds = [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
             if ~isempty(reliableTracks)
                                                    % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                                                    % Get ids.
                ids = int32([reliableTracks(:).id]);
                                                     % Create labels for objects indicating the ones for which we display the predicted rather than the actual location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {'pre'};
                labels = strcat(labels, isPredicted);
                                                                     % Draw the objects on the frame.
               
              % Fishstruct{1}=struct('midline',[],'boundary',[]);
               for k1=1:length(ids)
                      Fishstruct(k1)=struct('midline',zeros(length(reliableTracks(k1).midlineX),2),'boundary',zeros(length(reliableTracks(k1).midlineX),2));
                      Fishstruct(k1)=struct('midline',[reliableTracks(k1).midlineX, reliableTracks(k1).midlineY],'boundary',[reliableTracks(k1).boundaryX, reliableTracks(k1).boundaryY]);
               end
                
                region =bboxes;
                region(:, 3) = 5;
                frame = insertObjectAnnotation(frame, 'circle',  region, labels,'Color', 'blue');               
               
                sprintf('==========================================')
               for k1=1:length(bboxes)
                  fprintf('Center of fish: ID=%d,     X=%0.2f,     Y= %0.2f\n',k1, bboxes(k1,1), bboxes(k1,2));
               end
             end
             
             frameCount = frameCount+1;
              frameData(frameCount) = struct('fishesData',Fishstruct);
        end
        
                                                            % Display the mask and the frame.
        %obj.maskPlayer.step(mask);        
        obj.videoPlayer.step(frame);
    end

displayEndOfDemoMessage(mfilename)
end



