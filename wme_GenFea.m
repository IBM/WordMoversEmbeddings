% This script generates text embedding for a p.d. text kernel constructed
% from data-dependent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents.
% Here, we need to compute ground distance for every pair of unique words 
% in order to compute WMD. This is not efficient since there are a lot of 
% redundent computations. 
%
% Author: Lingfei Wu
% Date: 11/28/2018

function [Train,Test,Runtime] = wme_GenFea(Data,...
    gamma,R,DMin,DMax,dataSplit,...
    val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc,...
    randdoc_scheme,wordweight_scheme)

    if size(Data.TR,1) == 1
        dataSplit = 1;
    end
    train_words = Data.words(Data.TR(dataSplit,:));
    train_BOW_X = Data.BOW_X(Data.TR(dataSplit,:));
    train_X = Data.X(Data.TR(dataSplit,:));
    train_Y = Data.Y(Data.TR(dataSplit,:));
    test_words = Data.words(Data.TE(dataSplit,:));
    test_BOW_X = Data.BOW_X(Data.TE(dataSplit,:));
    test_X = Data.X(Data.TE(dataSplit,:));
    test_Y = Data.Y(Data.TE(dataSplit,:));

    % get nbow and tf-idf weights
    train_NBOW_X = nbow_X_allDoc(Data.TR(dataSplit,:));
    train_IDF_X = idf_X_allDoc(Data.TR(dataSplit,:));
    train_TFIDF_X = tf_idf_X_allDoc(Data.TR(dataSplit,:));
    test_NBOW_X = nbow_X_allDoc(Data.TE(dataSplit,:));
    test_TFIDF_X = tf_idf_X_allDoc(Data.TE(dataSplit,:));

    % generate random features based on emd distance between original texts
    % and random texts where random words are sampled in R^d word space
    timer_start = tic;
    rng('default')
    if randdoc_scheme == 1 
        % Method 1: RF features - uniform distribution. Generate random 
        % features based on emd distance between original documents and 
        % random documents where random words are sampled in R^d word space
        timer_start = tic;
        sample_X = cell(1,R);
        sample_weight_X = cell(1,R);
        for i=1:R
            D = randi([DMin,DMax],1);
%             sample_X{i} = randn(d,D)./sigma; % gaussian
            sample_X{i} = val_min+(val_max-val_min)*(rand(d,D)); % 
            % uniform normalize random word vector into an unit vector 
            % to be consistent with pre-trained words in word2vector space
            for ii=1:D
                sample_X{i}(:,ii) = sample_X{i}(:,ii)/norm(sample_X{i}(:,ii));
            end
            sample_weight_X{i} = ones(1,D); % uniform frequence for random word
        end
    end

    if wordweight_scheme == 1 % use NBOW
        train_weight_X = train_NBOW_X;
        test_weight_X = test_NBOW_X;
    elseif wordweight_scheme == 2 % use TFIDF
        train_weight_X = train_TFIDF_X;
        test_weight_X = test_TFIDF_X;
    end
    [trainFeaX_random, train_emd_time] = wmd_dist(train_X,train_weight_X,...
        sample_X,sample_weight_X,gamma);
    fprintf('Finish computing trainFeaX \n');
    [testFeaX_random, test_emd_time] = wmd_dist(test_X,test_weight_X,...
        sample_X,sample_weight_X,gamma);
    fprintf('Finish computing testFeaX \n');
    trainFeaX_random = trainFeaX_random/sqrt(R); 
    testFeaX_random = testFeaX_random/sqrt(R);
    Train = [train_Y', trainFeaX_random];
    Test = [test_Y', testFeaX_random];
    telapsed_random_fea_gen = toc(timer_start);
    
    % Note: real_total_end_time is the real total time, including both emd
    % and ground distance, of generating both train and test features using 
    % multithreads. user_emd_time is the real time that accounts for 
    % computation of emd with one thread. 
    Runtime.real_total_emd_time = telapsed_random_fea_gen;
    Runtime.user_emd_time = train_emd_time + test_emd_time;
    Runtime.user_train_emd_time = train_emd_time;
    Runtime.user_test_emd_time = test_emd_time;
end
