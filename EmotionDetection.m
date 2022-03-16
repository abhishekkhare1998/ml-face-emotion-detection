
dataset_folder = dir('.\upload_dataset\');
% image dimensions obtained from the images.
image_size = 193*162;

% get image names
[normal_face_img, smiling_face_img] = load_img_names(dataset_folder);

% Randomly Read 100 normal and smiling images from the Data
read_rand = randperm(size(normal_face_img,1),100);
test_images = setdiff(1:size(normal_face_img,1),read_rand);
normal_images = zeros(image_size,100);
smiling_images = zeros(image_size,100);

for i = 1:100
    % Use read_rand to randomly read 100 images for normal and smiling
    read_temp = imread(strcat('./upload_dataset/',normal_face_img(read_rand(1,i),1)));
    normal_images(:,i) = reshape(read_temp,size(read_temp,1)*size(read_temp,2),1);

    read_temp = imread(strcat('./upload_dataset/',smiling_face_img(read_rand(1,i),1)));
    smiling_images(:,i) = reshape(read_temp,size(read_temp,1)*size(read_temp,2),1);
end
 

% mean val of images
Avg_img_normal = mean(normal_images,2);
Avg_img_smiling = mean(smiling_images,2);

% calculate covariance matrix for set A and B
zero_mean_img_A = zeros(image_size,100);
zero_mean_img_B = zeros(image_size,100);
for i=1:100
    zero_mean_img_A(:,i) = normal_images(:,i) - Avg_img_normal;
    zero_mean_img_B(:,i) = smiling_images(:,i) - Avg_img_smiling;
end

Cov_A = zero_mean_img_A'*zero_mean_img_A;
Cov_B = zero_mean_img_B'*zero_mean_img_B;

% Plot SVD of the Cov_A&B Matrix
figure("Name","SVD of the neutral images")
plot(svd(Cov_A))
figure("Name","SVD of the smiling images")
plot(svd(Cov_B))

MSE_A = [];
MSE_B = [];
MSE_Test_A = [];
MSE_Test_B = [];
K = 100;

test_img_num = test_images(1,randi([1 71]));
% rand test image from set A
nameA = normal_face_img(test_img_num);
img_A_test = imread(strcat('./upload_dataset/',nameA));
res_image_A_test = double(reshape(img_A_test, image_size,1));

% rand test image from set B
nameB = smiling_face_img(test_img_num);
img_B_test = imread(strcat('./upload_dataset/',nameB));
res_image_B_test = double(reshape(img_B_test, image_size,1));

for k = 0:10:100
    
    % Compute k principle components for training set A 
    [eigen_vec_norm_A, eigen_vec_weighted_A] = perform_PCA(Cov_A,k,zero_mean_img_A);

    % Reconstruction of the Image from Training Set A
    Reconstructed_Image_A = sum((eigen_vec_weighted_A(:,5) .* eigen_vec_norm_A')',2);

    if (k == 60 || k == 80)
    figure("Name","Reconstructed Image A"),imshow(reshape(Reconstructed_Image_A+Avg_img_normal,193,162),[]);
    figure("Name","Original Image A"), imshow(imread(strcat('./upload_dataset/',normal_face_img(read_rand(1,5),1))));
    end

    err = (norm(Reconstructed_Image_A - zero_mean_img_A(:,5)).^2)/31266;
    MSE_A = [MSE_A err];

    % Compute PCs for Training Set Images B with k as the number of PCs
    [eigen_vec_norm_B, eigen_vec_weighted_B] = perform_PCA(Cov_B,k,zero_mean_img_B);
    
    % Reconstruction of the Image from Training Set B
    Reconstructed_Image_B = sum((eigen_vec_weighted_B(:,5) .* eigen_vec_norm_B')',2);
    
    
    if (k == 60 || k == 80)
    figure("Name","Reconstructed Image B"), imshow(reshape(Reconstructed_Image_B+Avg_img_smiling,193,162),[])
    figure("Name","Original Image B"), imshow(imread(strcat('./upload_dataset/',smiling_face_img(read_rand(1,5),1))));
    end 
    err = (norm(Reconstructed_Image_B - zero_mean_img_B(:,5)).^2)/31266;
    MSE_B = [MSE_B err];

    % Reconstruction of Neutral Image not from the training Set
    Residual_image_test_A = res_image_A_test - Avg_img_normal;
    Weights_Eig_Vec_Test_Im_A = zeros(k,1);
    for i = 1:k
        Weights_Eig_Vec_Test_Im_A(i,1) = eigen_vec_norm_A(:,i)'*Residual_image_test_A;
    end

    Reconstructed_Test_Im_A = sum((Weights_Eig_Vec_Test_Im_A .* eigen_vec_norm_A')',2);
    if (k == 60 || k == 80)
    figure("Name","Reconstructed Test Image A"), imshow(reshape(Reconstructed_Test_Im_A+Avg_img_normal,193,162),[])
    figure("Name","Original Test Image A"), imshow(img_A_test);
    end

    err = (norm(Reconstructed_Test_Im_A - Residual_image_test_A).^2)/31266;
    MSE_Test_A = [MSE_Test_A err];

    % Reconstruction of Similing Image not from the training Set
    Residual_image_test_B = res_image_B_test - Avg_img_smiling;
    Weights_Eig_Vec_Test_Im_B = zeros(k,1);
    for i = 1:k
        Weights_Eig_Vec_Test_Im_B(i,1) = eigen_vec_norm_B(:,i)'*Residual_image_test_B;
    end

    Reconstructed_Test_Im_B = sum((Weights_Eig_Vec_Test_Im_B .* eigen_vec_norm_B')',2);
    if (k == 60 || k == 80)
    figure("Name","Reconstructed Test Image B"), imshow(reshape(Reconstructed_Test_Im_B+Avg_img_smiling,193,162),[])
    figure("Name","Original Test Image B"), imshow(img_B_test);
    end
    err = (norm(Reconstructed_Test_Im_B - Residual_image_test_B).^2)/31266;
    MSE_Test_B = [MSE_Test_B err];

end

% Display MSE of reconstructed images (test+training) vs number of principle components
ii = 0:10:100;
figure("Name","MSE of reconstructed normal (set A) images wrt num of PCs")
plot(ii,MSE_A)
figure("Name","MSE of reconstructed smiling images (set B) wrt num of PCs")
plot(ii,MSE_B)
figure("Name","MSE of reconstructed Test Images for set A wrt num of PCs")
plot(ii,MSE_Test_A)
figure("Name","MSE of reconstructed Test Images for set B wrt num of PCs")
plot(ii,MSE_Test_B)

%%%% part 5 %%%%%%
test_image_num = randperm(71,60)';
image_A_name = normal_face_img(test_image_num,1);
image_B_name = smiling_face_img(test_image_num,1);
test_images_overall = [image_A_name;image_B_name];
prediction_list = [];
num_of_correct_predictions = 0;
correct_classification_A = 0;
correct_classification_B = 0;
incorrect_classification_A = 0;
incorrect_classification_B = 0;
incorrect_classification_names = [];

for i = 1:120
    incorrect_class_name = "";
    if endsWith(test_images_overall(i,1),'a.jpg')
        actual_class = 1;
    end
    if endsWith(test_images_overall(i,1),'b.jpg')
        actual_class = 0;
    end

    read_img = imread(strcat('./upload_dataset/',test_images_overall(i,1)));
    test_images_class(:,i) = reshape(read_img, image_size,1);

    zero_mean_img_A = double(test_images_class(:,i)) - Avg_img_normal;
    [MSE_image_A, reconstructed_for_space_A] = recon_and_mse_for_classification(zero_mean_img_A, k, image_size, eigen_vec_norm_A);

    zero_mean_img_B = double(test_images_class(:,i)) - Avg_img_smiling;
    [MSE_image_B, reconstructed_for_space_B] = recon_and_mse_for_classification(zero_mean_img_B, k, image_size, eigen_vec_norm_B);

    current_name = test_images_overall(i,1);
    len_of_name = strlength(current_name);
    if(MSE_image_A < MSE_image_B)
        current_prediction =  strcat(extractBetween(current_name,1,len_of_name-5),'a.jpg');
    else
        current_prediction =  strcat(extractBetween(current_name,1,len_of_name-5),'b.jpg');
    end
    prediction_list = [prediction_list;current_prediction];
    if current_prediction == current_name
        num_of_correct_predictions = num_of_correct_predictions + 1;
    end

    %%%% comparing results using MSE of projection on eigenspaces A and B
    %%%% respectively
    if actual_class == 1 && (MSE_image_A < MSE_image_B)
        correct_classification_A = correct_classification_A + 1;
    end

    if actual_class == 1 && (MSE_image_A > MSE_image_B)
        incorrect_classification_A = incorrect_classification_A + 1;
        incorrect_class_name = current_name;
    end

    if actual_class == 0 && (MSE_image_B < MSE_image_A)
        correct_classification_B = correct_classification_B + 1;
    end

    if actual_class == 0 && (MSE_image_B > MSE_image_A)
        incorrect_classification_B = incorrect_classification_B + 1;
        incorrect_class_name = current_name;
    end

    if incorrect_class_name == ""
    else
    incorrect_classification_names = [incorrect_classification_names; incorrect_class_name];
    end

end
total_accuracy = (double(num_of_correct_predictions)/120) *100;
neutral_face_accuracy = (double(correct_classification_A)/60) *100;
smiling_face_accuracy = (double(correct_classification_B)/60) *100;
disp(incorrect_classification_names)


%%%%%% functions %%%%%%%%%%%%


function [eigen_vec_norm, eigen_vec_weighted] = perform_PCA(Cov_Matrix,Num_Of_PCs,zero_mean_img_set)
    size1 = 100;
    [eigenvectors,temp] = eig(Cov_Matrix);
    eigenval = eig(Cov_Matrix);
    [ordered_eigenval, Index_of_Eigen_Values] = sort(eigenval,'descend');
    maxEigenVec = zeros(size(eigenvectors,1),Num_Of_PCs);
    
    for i=1:Num_Of_PCs
        maxEigenVec(:,i) = eigenvectors(:,Index_of_Eigen_Values(i,1));
    end
    
    True_Max_Eigen_Vectors = zero_mean_img_set*maxEigenVec;
    
    eigen_vec_weighted = zeros(Num_Of_PCs,size1);
    eigen_vec_norm = normc(True_Max_Eigen_Vectors);
    for j= 1:size1
        for i=1:Num_Of_PCs
            eigen_vec_weighted(i,j) = eigen_vec_norm(:,i)'*zero_mean_img_set(:,j);
        end
    end

end

function [normal_face_img, smiling_face_img] = load_img_names(dataset_folder)
normal_face_img = [];
smiling_face_img = [];
for i = 3:size(dataset_folder,1)
    str1 = convertCharsToStrings(dataset_folder(i,1).name);
    if(endsWith(str1,'a.jpg'))
        normal_face_img = [normal_face_img; str1];
    end
    if(endsWith(str1,'b.jpg'))
        smiling_face_img = [smiling_face_img; str1];
    end    
end
end

function [MSE_image, recon_test_img] = recon_and_mse_for_classification(zero_mean_img, k, image_size, eigen_vec_norm)

eigenvec_weights = zeros(k,1);
for ii = 1:k
    eigenvec_weights(ii,1) = eigen_vec_norm(:,ii)'*zero_mean_img;
end
recon_test_img = sum((eigenvec_weights .* eigen_vec_norm')',2);
MSE_image = [];
err = (norm(recon_test_img - zero_mean_img).^2)/image_size;
MSE_image = [MSE_image err];
%figure("Name","Reconstructed FEC Image B"), imshow(reshape(recon_test_img+zero_mean_img,193,162),[])
end