cluster_size = 500;

%Perform K-means
[idx,centroids] = kmeans(vectors, cluster_size,'Display','iter');


%Generate histograms per image
noOfImages = 0;

for i=1:length(classes)
    
    noOfImages = noOfImages + length(datalog{i});
    
end

histograms = zeros(noOfImages,cluster_size);
labels = zeros(noOfImages, 1);

featureOffset = 1;
imgOffset = 0;

for i=1:length(classes)
   
    featureSizesPerImage = datalog{i};
    
    %Iterate all over the images, the length = number of images in class i
    for j = 1:length(featureSizesPerImage)

        noOfFeatures = featureSizesPerImage(j);        
        
        imgIdx = j + imgOffset;
        
        for n = featureOffset:featureOffset+noOfFeatures-1
            
            histograms(imgIdx,idx(n)) = histograms(imgIdx,idx(n)) + 1; %Update histogram per feature
            
        end
        
        labels(imgIdx) = i; %Set label for each image
        
        
        featureOffset = featureOffset+noOfFeatures;
               
    end
    
    imgOffset = imgOffset + length(featureSizesPerImage);
    
    
end
