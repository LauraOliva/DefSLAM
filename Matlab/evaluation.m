close all; clear all;

l = [];

%% Load files from folder
path ='/home/laura/ExperimentsResults/DefSLAM/Hamlyn/abdominal/';
name = "abdominal";
l = [l name];

% Timestamps = ni, strat=200
start=200;

% ScaleVariation: scale drift -> scale values of each keyframe: timestamp, scale value
scale_filename = path+"/ScaleVariation.txt";
% Matches: timestamp, #inliers, #outliers, #mappoints
matches_filename = path+"/Matches.txt";
% MapPointUsage -> empty
map_points_filename = path+"/MapPointsUsage.txt";

%% Scale
data_scale = dlmread(scale_filename);
figure(1);
hold on;
plot(data_scale(:,1), data_scale(:,2)/data_scale(1,2));
title("Scale Variation");
ylabel("Scale drift");
xlabel("#Frame");

legend(l);

nimages=start+length(data_scale)-1;

%% Matches
data_matches = dlmread(matches_filename);
figure(2);
% Inliers
plot(data_matches(:,1), data_matches(:,2), '-r');
hold on;
% Outliers
plot(data_matches(:,1), data_matches(:,3), '-g');
hold on;
% Map Points
plot(data_matches(:,1), data_matches(:,4), '-b');
grid on;

title("Matches");
ylabel("#Points");
xlabel("#Frame");

legend(["Inliers", "Outliers", "Map Points"]);


%% Matches - ratio
ratio_matches = data_matches(:,2) ./ data_matches(:,4);
figure(3);
plot(ratio_matches);
grid on;

title("Matches - ratio");
ylabel("Fraction of matched map points");
xlabel("#Frame");
ylim([0 1]')


%% RMS error - GT
data_error_gt = [];
for i = start:1:nimages
    % ErrorAnglso: angular RMS between the angle estimated by the IsoNRSfM and the angle error agter fitting a surface to the normals estimated.
    
    % ErrorAngSfN
    
    % ErrorGTs: we recover the scale per frame and then we estimate the RMS error between both the scaled surface and the GT surface.
    % One file per keyframe, one line per map point
    % Error of the point in that KF
    error_gt_filename = path + "ErrorGTs" + num2str(i, '%05d') + ".txt"; 
    %std::cout << "Mean Error Surf : " << acc * invc << " " << Error.size() << " "
    %          << posMono_.size() << std::endl;
    % acc = sum(errors)
    % inv = 1 / num_map_points

    data_error_gt_i = dlmread(error_gt_filename);
    acc = sum(data_error_gt_i);
    data_error_gt = [data_error_gt; acc/length(data_error_gt_i)];

end

figure(4);
plot(data_error_gt);
title("GT error");
ylabel("3D RMS error(mm)");
xlabel("#Frame");

%% Evolution of the error

% estimateAngleErrorAndScale. This method estimates the 3D groundtruth of the
% surface estimated for this keyframe. It use the keypoints of the left image 
% with a normal and search for estimates the 3D in the right image. It uses the 
% pcl library to determinate the normals of the point cloud and compares them 
% with the estimated by the NRSfM and the SfN. NRSfM tends to be quite noisy.

% id of the kf: 
%   if the system needs a new template -> current KF
%   else -> From the observed points we select the keyframe with the highest number of observed map points.

% New kf every 10 frames

% std::cout << "Mean angle Iso Error Keyframe : " << std::endl
% << "min : "
% << *std::min_element(ErrorAngleIso.begin(), ErrorAngleIso.end())
% << std::endl
% << "max : "
% << *std::max_element(ErrorAngleIso.begin(), ErrorAngleIso.end())
% << std::endl
% << "median : " << ErrorAngleIso[ErrorAngleIso.size() / 2]
% << std::endl
% << sumIso / ((double)ErrorAngleIso.size()) << " " << std::endl;


angiso_filenames = dir(strcat(path,'ErrorAngIso*'));
nframes = length(angiso_filenames);
angiso_data = [];
angsfn_data = [];

for i = 0:1:nframes-1
    angiso_filename = dir(strcat(path,'ErrorAngIso*-', num2str(i), '.*'));
    angsfn_filename = dir(strcat(path,'ErrorAngSfN*-', num2str(i), '.*'));

    angiso_data_i = dlmread(strcat(path, angiso_filename.name));
    angsfn_data_i = dlmread(strcat(path, angsfn_filename.name));

    angiso_data = [angiso_data; mean(angiso_data_i)];
    angsfn_data = [angsfn_data; mean(angsfn_data_i)];
end

figure(5);
frames = [0:10:(nframes-1)*10];
stairs(frames, angiso_data, '-r');
hold on;
stairs(frames, angsfn_data, '-b');
grid on; 

ylabel('Angle Error (degrees)');
xlabel('#Frame');
legend(["Angle Error Iso", "Angle Error SfN"]);



