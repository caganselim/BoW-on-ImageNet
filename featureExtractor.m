run('vlfeat/toolbox/vl_setup')

%Parameters
%SIFT
peak_thresh = 2;
edge_thresh = 5;

%Cluster size
cluster_size = 128;

%End of Parameters

container = zeros(130,4000000);
datalog = cell(10,1);

classes = ["n02123159","n02676566","n02773838","n03179701","n03255030","n03445777","n03642806","n03792782","n04037443","n04555897"];

offset = 1;

for i=1:length(classes)
    
    %Find all files inside the class
    classPath = strcat('cropped/', classes(i));
    searchQuery = strcat(classPath,"\**/*.JPEG");
    imgs = dir(searchQuery);
    count = zeros(length(imgs),1);
    
    for j=1:length(imgs)
       
        imgname =  imgs(j).name;
        imaddr = strcat(classPath, '/' , imgname);
        origIm = imread(imaddr);
        
        if  size(origIm,3)==3
            I = single(rgb2gray(origIm));
            
        else
            
            I = single(origIm);
            
        end


        [f,d] = vl_dsift(I, 'size', 20, 'step', 10);
        
        
        noOfFeatures = size(d,2);
        
        disp(imaddr);
        disp(noOfFeatures);
        
        count(j) = noOfFeatures;
      
        container(1,offset:(offset+noOfFeatures-1)) = i; %save the class
         
        imgid = extractBetween(imgname,'_','.');
        imgid = str2double(imgid{1,1});
        
        container(2,offset:(offset+noOfFeatures-1)) = imgid;
        
        container(3:130,offset:(offset+noOfFeatures-1)) = double(d); %save the feature vector
        
        offset = offset + noOfFeatures;

    end
    
    datalog{i} = count;
end

%Wrap Up
container = container(:,1:offset);
container = transpose(container);
vectors = container(:,3:130);

disp(size(vectors))


% imshow(origIm);
% h1 = vl_plotframe(f) ;
% h2 = vl_plotframe(f) ;
% set(h1,'color','k','linewidth',3) ;
% set(h2,'color','y','linewidth',2) ;
% h3 = vl_plotsiftdescriptor(d,f);


